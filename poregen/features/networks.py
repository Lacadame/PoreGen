"""
Clone of PoreSpy regions_to_network function, but without tqdm
"""

import logging
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball
from skimage.segmentation import relabel_sequential, find_boundaries
from edt import edt

from .surface_area import extend_slice, region_surface_areas, region_interface_areas, region_volumes
from .porosimetry import borders


__all__ = [
    "regions_to_network",
]


logger = logging.getLogger(__name__)


def regions_to_network(regions, phases=None, voxel_size=1, accuracy='standard'):  # noqa: C901
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    regions : ndarray
        An image of the material partitioned into individual regions.
        Zeros in this image are ignored.
    phases : ndarray, optional
        An image indicating to which phase each voxel belongs. The returned
        network contains a 'pore.phase' array with the corresponding value.
        If not given a value of 1 is assigned to every pore.
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    accuracy : string
        Controls how accurately certain properties are calculated. Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels.  This is *much* faster but does not properly account
            for the rough, voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method.  These are substantially
            slower but better account for the voxelated nature of the images.

    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').

    Notes
    -----
    The meaning of each of the values returned in ``net`` are outlined below:

    'pore.region_label'
        The region labels corresponding to the watershed extraction. The
        pore indices and regions labels will be offset by 1, so pore 0
        will be region 1.
    'throat.conns'
        An *Nt-by-2* array indicating which pores are connected to each other
    'pore.region_label'
        Mapping of regions in the watershed segmentation to pores in the
        network
    'pore.local_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the pore region in isolation
    'pore.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.geometric_centroid'
        The center of mass of the pore region as calculated by
        ``skimage.measure.center_of_mass``
    'throat.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.region_volume'
        The volume of the pore region computed by summing the voxels
    'pore.volume'
        The volume of the pore found by as volume of a mesh obtained from the
        marching cubes algorithm
    'pore.surface_area'
        The surface area of the pore region as calculated by either counting
        voxels or using the fast marching method to generate a tetramesh (if
        ``accuracy`` is set to ``'high'``.)
    'throat.cross_sectional_area'
        The cross-sectional area of the throat found by either counting
        voxels or using the fast marching method to generate a tetramesh (if
        ``accuracy`` is set to ``'high'``.)
    'throat.perimeter'
        The perimeter of the throat found by counting voxels on the edge of
        the region defined by the intersection of two regions.
    'pore.inscribed_diameter'
        The diameter of the largest sphere inscribed in the pore region. This
        is found as the maximum of the distance transform on the region in
        isolation.
    'pore.extended_diameter'
        The diamter of the largest sphere inscribed in overal image, which
        can extend outside the pore region. This is found as the local maximum
        of the distance transform on the full image.
    'throat.inscribed_diameter'
        The diameter of the largest sphere inscribed in the throat.  This
        is found as the local maximum of the distance transform in the area
        where to regions meet.
    'throat.total_length'
        The length between pore centered via the throat center
    'throat.direct_length'
        The length between two pore centers on a straight line between them
        that does not pass through the throat centroid.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/regions_to_network.html>`_
    to view online example.

    """
    logger.info('Extracting pore/throat information')

    im = make_contiguous(regions)
    struc_elem = disk if im.ndim == 2 else ball
    voxel_size = float(voxel_size)
    if phases is None:
        phases = (im > 0).astype(int)
    if im.size != phases.size:
        raise Exception('regions and phase are different sizes, probably ' +
                        'because boundary regions were not added to phases')
    dt = np.zeros_like(phases, dtype="float32")  # since edt returns float32
    for i in np.unique(phases[phases.nonzero()]):
        dt += edt(phases == i)

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)

    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    Np = np.size(Ps)
    p_coords_cm = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, im.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_label = np.zeros((Np, ), dtype=int)
    p_area_surf = np.zeros((Np, ), dtype=int)
    p_phase = np.zeros((Np, ), dtype=int)
    # The number of throats is not known at the start, so lists are used
    # which can be dynamically resized more easily.
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []

    # Start extracting size information for pores and throats
    msg = "Extracting pore and throat properties"
    for i in Ps:
        pore = i - 1
        if slices[pore] is None:
            continue
        s = extend_slice(slices[pore], im.shape)
        sub_im = im[s]
        sub_dt = dt[s]
        pore_im = sub_im == i
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = edt(padded_mask)
        s_offset = np.array([i.start for i in s])
        p_label[pore] = i
        p_coords_cm[pore, :] = spim.center_of_mass(pore_im) + s_offset
        temp = np.vstack(np.where(pore_dt == pore_dt.max()))[:, 0]
        p_coords_dt[pore, :] = temp + s_offset
        p_phase[pore] = (phases[s]*pore_im).max()
        temp = np.vstack(np.where(sub_dt == sub_dt.max()))[:, 0]
        p_coords_dt_global[pore, :] = temp + s_offset
        p_volume[pore] = np.sum(pore_im, dtype=np.int64)
        p_dia_local[pore] = 2*np.amax(pore_dt)
        p_dia_global[pore] = 2*np.amax(sub_dt)
        # The following is overwritten if accuracy is set to 'high'
        p_area_surf[pore] = np.sum(pore_dt == 1, dtype=np.int64)
        im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im
        Pn = np.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = np.where(im_w_throats == (j + 1))
                t_dia_inscribed.append(2*np.amax(sub_dt[vx]))
                # The following is overwritten if accuracy is set to 'high'
                t_perimeter.append(np.sum(sub_dt[vx] < 2, dtype=np.int64))
                # The following is overwritten if accuracy is set to 'high'
                t_area.append(np.size(vx[0]))
                p_area_surf[pore] -= np.size(vx[0])
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
                t_coords.append(tuple([t_inds[k][temp] for k in range(im.ndim)]))

    # Clean up values
    p_coords = p_coords_cm
    Nt = len(t_dia_inscribed)  # Get number of throats
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords_cm.T, np.zeros((Np, )))).T
        t_coords = np.vstack((np.array(t_coords).T, np.zeros((Nt, )))).T

    net = {}
    ND = im.ndim
    # Define all the fundamental stuff
    net['throat.conns'] = np.array(t_conns)
    net['pore.coords'] = np.array(p_coords)*voxel_size
    net['pore.all'] = np.ones_like(net['pore.coords'][:, 0], dtype=bool)
    net['throat.all'] = np.ones_like(net['throat.conns'][:, 0], dtype=bool)
    net['pore.region_label'] = np.array(p_label)
    net['pore.phase'] = np.array(p_phase, dtype=int)
    net['throat.phases'] = net['pore.phase'][net['throat.conns']]
    V = np.copy(p_volume)*(voxel_size**ND)
    net['pore.region_volume'] = V  # This will be an area if image is 2D
    f = 3/4 if ND == 3 else 1.0
    net['pore.equivalent_diameter'] = 2*(V/np.pi * f)**(1/ND)
    # Extract the geometric stuff
    net['pore.local_peak'] = np.copy(p_coords_dt)*voxel_size
    net['pore.global_peak'] = np.copy(p_coords_dt_global)*voxel_size
    net['pore.geometric_centroid'] = np.copy(p_coords_cm)*voxel_size
    net['throat.global_peak'] = np.array(t_coords)*voxel_size
    net['pore.inscribed_diameter'] = np.copy(p_dia_local)*voxel_size
    net['pore.extended_diameter'] = np.copy(p_dia_global)*voxel_size
    net['throat.inscribed_diameter'] = np.array(t_dia_inscribed)*voxel_size
    P12 = net['throat.conns']
    PT1 = np.sqrt(np.sum(((p_coords[P12[:, 0]]-t_coords)*voxel_size)**2,
                         axis=1))
    PT2 = np.sqrt(np.sum(((p_coords[P12[:, 1]]-t_coords)*voxel_size)**2,
                         axis=1))
    net['throat.total_length'] = PT1 + PT2
    PT1 = PT1-p_dia_local[P12[:, 0]]/2*voxel_size
    PT2 = PT2-p_dia_local[P12[:, 1]]/2*voxel_size
    dist = (p_coords[P12[:, 0]] - p_coords[P12[:, 1]])*voxel_size
    net['throat.direct_length'] = np.sqrt(np.sum(dist**2, axis=1, dtype=np.int64))
    net['throat.perimeter'] = np.array(t_perimeter)*voxel_size
    if (accuracy == 'high') and (im.ndim == 2):
        msg = "accuracy='high' only available in 3D, reverting to 'standard'"
        logger.warning(msg)
        accuracy = 'standard'
    if (accuracy == 'high'):
        net['pore.volume'] = region_volumes(regions=im, mode='marching_cubes')
        areas = region_surface_areas(regions=im, voxel_size=voxel_size)
        net['pore.surface_area'] = areas
        interface_area = region_interface_areas(regions=im, areas=areas,
                                                voxel_size=voxel_size)
        A = interface_area.area
        net['throat.cross_sectional_area'] = A
        net['throat.equivalent_diameter'] = (4*A/np.pi)**(1/2)
    else:
        net['pore.volume'] = np.copy(p_volume)*(voxel_size**ND)
        net['pore.surface_area'] = np.copy(p_area_surf)*(voxel_size**2)
        A = np.array(t_area)*(voxel_size**2)
        net['throat.cross_sectional_area'] = A
        net['throat.equivalent_diameter'] = (4*A/np.pi)**(1/2)

    return net


def make_contiguous(im, mode='keep_zeros'):
    r"""
    Take an image with arbitrary greyscale values and adjust them to ensure
    all values fall in a contiguous range starting at 0.

    Parameters
    ----------
    im : array_like
        An ND array containing greyscale values
    mode : string
        Controls how the ranking is applied in the presence of numbers less
        than or equal to 0.

        'keep_zeros'
            (default) Voxels equal to 0 remain 0, and all other
            numbers are ranked starting at 1, include negative numbers,
            so [-1, 0, 4] becomes [1, 0, 2]

        'symmetric'
            Negative and positive voxels are ranked based on their
            respective distances to 0, so [-4, -1, 0, 5] becomes
            [-2, -1, 0, 1]

        'clipped'
            Voxels less than or equal to 0 are set to 0, while
            all other numbers are ranked starting at 1, so [-3, 0, 2]
            becomes [0, 0, 1].

        'none'
            Voxels are ranked such that the smallest or most
            negative number becomes 1, so [-4, 2, 0] becomes [1, 3, 2].
            This is equivalent to calling ``scipy.stats.rankdata`` directly,
            and reshaping the result to match ``im``.

    Returns
    -------
    image : ndarray
        An ndarray the same size as ``im`` but with all values in contiguous
        order.

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> im = np.array([[0, 2, 9], [6, 8, 3]])
    >>> im = ps.tools.make_contiguous(im)
    >>> print(im)
    [[0 1 5]
     [3 4 2]]

    `Click here
    <https://porespy.org/examples/tools/reference/make_contiguous.html>`_
    to view online example.

    """
    # This is a very simple version using relabel_sequential
    im = np.array(im)
    if mode == 'none':
        im = im + np.abs(np.min(im)) + 1
        im_new = relabel_sequential(im)[0]
    if mode == 'keep_zeros':
        mask = im == 0
        im = im + np.abs(np.min(im)) + 1
        im[mask] = 0
        im_new = relabel_sequential(im)[0]
    if mode == 'clipped':
        mask = im <= 0
        im[mask] = 0
        im_new = relabel_sequential(im)[0]
    if mode == 'symmetric':
        mask = im < 0
        im_neg = relabel_sequential(-im*mask)[0]
        mask = im >= 0
        im_pos = relabel_sequential(im*mask)[0]
        im_new = im_pos - im_neg
    return im_new


def label_phases(
        network,
        alias={1: 'void', 2: 'solid'}):
    r"""
    Create pore and throat labels based on 'pore.phase' values

    Parameters
    ----------
    network : dict
        The network stored as a dictionary as returned from the
        ``regions_to_network`` function
    alias : dict
        A mapping between integer values in 'pore.phase' and string labels.
        The default is ``{1: 'void', 2: 'solid'}`` which will result in the
        labels ``'pore.void'`` and ``'pore.solid'``, as well as
        ``'throat.solid_void'``, ``'throat.solid_solid'``, and
        ``'throat.void_void'``.  The reverse labels are also added for
        convenience like ``throat.void_solid``.

    Returns
    -------
    network : dict
        The same ``network`` as passed in but with new boolean arrays added
        for the phase labels.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/label_phases.html>`_
    to view online example.

    """
    conns = network['throat.conns']
    for i in alias.keys():
        pore_i_hits = network['pore.phase'] == i
        network['pore.' + alias[i]] = pore_i_hits
        for j in alias.keys():
            pore_j_hits = network['pore.phase'] == j
            throat_hits = pore_i_hits[conns[:, 0]] * pore_j_hits[conns[:, 1]]
            throat_hits += pore_i_hits[conns[:, 1]] * pore_j_hits[conns[:, 0]]
            if np.any(throat_hits):
                name = 'throat.' + '_'.join([alias[i], alias[j]])
                if name not in network.keys():
                    network[name] = np.zeros_like(conns[:, 0], dtype=bool)
                network[name] += throat_hits
    return network


def label_boundaries(
        network,
        labels=[['left', 'right'], ['front', 'back'], ['top', 'bottom']],
        tol=1e-9):
    r"""
    Create boundary pore labels based on proximity to axis extrema

    Parameters
    ----------
    network : dict
        The network stored as a dictionary as returned from the
        ``regions_to_network`` function
    labels : list of lists
        A 3-element list, with each element containing a pair of strings
        indicating the label to apply to the beginning and end of each axis.
        The default is ``[['left', 'right'], ['front', 'back'],
        ['top', 'bottom']]`` which will apply the label ``'left'`` to all
        pores with the minimum x-coordinate, and ``'right'`` to the pores
        with the maximum x-coordinate, and so on.

    Returns
    -------
    network : dict
        The same ``network`` as passed in but with new boolean arrays added
        for the boundary labels.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/label_boundaries.html>`_
    to view online example.

    """
    crds = network['pore.coords']
    extents = [[crds[:, i].min(), crds[:, i].max()]
               for i in range(len(crds[0, :]))]
    network['pore.boundary'] = np.zeros_like(crds[:, 0], dtype=bool)
    for i, axis in enumerate(labels):
        for j, face in enumerate(axis):
            if face:
                hits = crds[:, i] == extents[i][j]
                network['pore.boundary'] += hits
                network['pore.' + labels[i][j]] = hits
    return network


def add_boundary_regions(regions, pad_width=3):
    r"""
    Add boundary regions on specified faces of an image

    Parameters
    ----------
    regions : ndarray
        An image containing labelled regions, such as a watershed segmentation
    pad_width : array_like
        Number of layers to add to the beginning and end of each axis. This
        argument is handled similarly to the ``pad_width`` in the ``np.pad``
        function. A scalar adds the same amount to the beginning and end of
        each axis. [A, B] adds A to the beginning of each axis and B to the
        ends.  [[A, B], ..., [C, D]] adds A to the beginning and B to the
        end of the first axis, and so on. The default is to add 3 voxels on
        both ends of each axis.  One exception is is [A, B, C] which A to
        the beginning and end of the first axis, and so on. This extra option
        is useful for putting 0 on some axes (i.e. [3, 0, 0]).

    Returns
    -------
    padded_regions : ndarray
        An image with new regions padded on each side of the specified
        width.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/add_boundary_regions.html>`_
    to view online example.

    """
    # Parse user specified padding
    faces = np.array(pad_width)
    if faces.size == 1:
        faces = np.array([[faces, faces]]*regions.ndim)
    elif faces.size == regions.ndim:
        faces = np.vstack([faces]*2)
    else:
        pass
    t = faces.max()
    mx = regions.max()
    # Put a border around each region so padded regions are isolated
    bd = find_boundaries(regions, connectivity=regions.ndim, mode='inner')
    # Pad by t in all directions, this will be trimmed down later
    face_regions = np.pad(regions*(~bd), pad_width=t, mode='edge')
    # Set corners to 0 so regions don't connect across faces
    edges = borders(shape=face_regions.shape, mode='edges', thickness=t)
    face_regions[edges] = 0
    # Extract a mask of just the faces
    mask = borders(shape=face_regions.shape, mode='faces', thickness=t)
    # Relabel regions on faces
    new_regions = spim.label(face_regions*mask)[0] + mx*(face_regions > 0)
    new_regions[~mask] = regions.flatten()
    # Trim image down to user specified size
    s = tuple([slice(t-ax[0], -(t-ax[1]) or None) for ax in faces])
    new_regions = new_regions[s]
    new_regions = make_contiguous(new_regions)
    return new_regions
