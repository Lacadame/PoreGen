"""
Copy of PoreSpy region_surface_areas functions and auxiliaries
"""


import numpy as np
import scipy.ndimage as spim
import skimage.morphology
import skimage.measure

from .porosimetry import ps_round
from poregen.utils import AttrDict


def surface_area_density(im, voxel_size=1.0):
    surface_area = region_surface_areas(im, voxel_size)
    volume = np.prod(im.shape) * voxel_size**3
    return surface_area / volume


def region_surface_areas(regions, voxel_size=1, strel=None):
    r"""
    Extract the surface area of each region in a labeled image.

    Optionally, it can also find the the interfacial area between all
    adjoining regions.

    Parameters
    ----------
    regions : ndarray
        An image of the pore space partitioned into individual pore regions.
        Note that zeros in the image will not be considered for area
        calculation.
    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1.
    strel : array_like
        The structuring element used to blur the region.  If not provided,
        then a spherical element (or disk) with radius 1 is used.  See the
        docstring for ``mesh_region`` for more details, as this argument is
        passed to there.

    Returns
    -------
    areas : list
        A list containing the surface area of each region, offset by 1, such
        that the surface area of region 1 is stored in element 0 of the list.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/region_surface_areas.html>`_
    to view online example.

    """
    im = regions
    if strel is None:
        strel = ps_round(1, im.ndim, smooth=False)
    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)
    # Initialize arrays
    Ps = np.arange(1, np.amax(im) + 1)
    sa = np.zeros_like(Ps, dtype=float)
    # Start extracting marching cube area from im
    for i in Ps:
        reg = i - 1
        if slices[reg] is not None:
            s = extend_slice(slices[reg], im.shape)
            sub_im = im[s]
            mask_im = sub_im == i
            mesh = mesh_region(region=mask_im, strel=strel)
            sa[reg] = mesh_surface_area(mesh)
    result = sa * voxel_size**2
    return result


def region_interface_areas(regions, areas, voxel_size=1, strel=None):
    r"""
    Calculate the interfacial area between all pairs of adjecent regions

    Parameters
    ----------
    regions : ndarray
        An image of the pore space partitioned into individual pore regions.
        Note that zeros in the image will not be considered for area
        calculation.
    areas : array_like
        A list containing the areas of each regions, as determined by
        ``region_surface_area``.  Note that the region number and list index
        are offset by 1, such that the area for region 1 is stored in
        ``areas[0]``.
    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1.
    strel : array_like
        The structuring element used to blur the region.  If not provided,
        then a spherical element (or disk) with radius 1 is used.  See the
        docstring for ``mesh_region`` for more details, as this argument is
        passed to there.

    Returns
    -------
    areas : Results object
        A custom object with the following data added as named attributes:

        'conns'
            An N-regions by 2 array with each row containing the region
            number of an adjacent pair of regions.  For instance, if
            ``conns[0, 0]`` is 0 and ``conns[0, 1]`` is 5, then row 0 of
            ``area`` contains the interfacial area shared by regions 0 and 5.

        'area'
            The area calculated for each pair of regions in ``conns``

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/region_interface_areas.html>`_
    to view online example.

    """
    im = regions
    ball = ps_round(1, im.ndim, smooth=False)
    if strel is None:
        strel = np.copy(ball)
    # Get 'slices' into im for each region
    slices = spim.find_objects(im)
    # Initialize arrays
    Ps = np.arange(1, np.amax(im) + 1)
    sa = np.zeros_like(Ps, dtype=float)
    sa_combined = []  # Difficult to preallocate since number of conns unknown
    cn = []
    # Start extracting area from im
    for i in Ps:
        reg = i - 1
        if slices[reg] is not None:
            s = extend_slice(slices[reg], im.shape)
            sub_im = im[s]
            mask_im = sub_im == i
            sa[reg] = areas[reg]
            im_w_throats = spim.binary_dilation(input=mask_im,
                                                structure=ball)
            im_w_throats = im_w_throats * sub_im
            Pn = np.unique(im_w_throats)[1:] - 1
            for j in Pn:
                if j > reg:
                    cn.append([reg, j])
                    merged_region = im[(min(slices[reg][0].start,
                                            slices[j][0].start)):
                                       max(slices[reg][0].stop,
                                           slices[j][0].stop),
                                       (min(slices[reg][1].start,
                                            slices[j][1].start)):
                                       max(slices[reg][1].stop,
                                           slices[j][1].stop)]
                    merged_region = ((merged_region == reg + 1)
                                     + (merged_region == j + 1))
                    mesh = mesh_region(region=merged_region, strel=strel)
                    sa_combined.append(mesh_surface_area(mesh))
    # Interfacial area calculation
    cn = np.array(cn)
    ia = 0.5 * (sa[cn[:, 0]] + sa[cn[:, 1]] - sa_combined)
    ia[ia <= 0] = 1
    result = AttrDict()
    result.conns = cn
    result.area = ia * voxel_size**2
    return result


def region_volumes(regions, mode='marching_cubes'):
    r"""
    Compute volume of each labelled region in an image

    Parameters
    ----------
    regions : ndarray
        An image with labelled regions
    mode : string
        Controls the method used. Options are:

        'marching_cubes' (default)
            Finds a mesh for each region using the marching cubes algorithm
            from ``scikit-image``, then finds the volume of the mesh using the
            ``trimesh`` package.

        'voxel'
            Calculates the region volume as the sum of voxels within each
            region.

    Returns
    -------
    volumes : ndarray
        An array of shape [N by 1] where N is the number of labelled regions
        in the image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/mesh_volumes.html>`_
    to view online example.

    """
    slices = spim.find_objects(regions)
    vols = np.zeros([len(slices), ])
    for i, s in enumerate(slices):
        region = regions[s] == (i + 1)
        if mode == 'marching_cubes':
            vols[i] = mesh_volume(region)
        elif mode.startswith('voxel'):
            vols[i] = region.sum(dtype=np.int64)
    return vols


def mesh_volume(region):
    r"""
    Compute the volume of a single region by meshing it

    Parameters
    ----------
    region : ndarray
        An image with a single region labelled as ``True`` (or > 0)

    Returns
    -------
    volume : float
        The volume of the region computed by applyuing the marching cubes
        algorithm to the region, then finding the mesh volume using the
        ``trimesh`` package.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/mesh_volume.html>`_
    to view online example.

    """
    try:
        from trimesh import Trimesh
    except ModuleNotFoundError:
        msg = 'The trimesh package can be installed with pip install trimesh'
        raise ModuleNotFoundError(msg)
    mc = mesh_region(region > 0)
    m = Trimesh(vertices=mc.verts, faces=mc.faces, vertex_normals=mc.norm)
    if m.is_watertight:
        vol = np.abs(m.volume)
    else:
        vol = np.nan
    return vol


def mesh_surface_area(mesh=None, verts=None, faces=None):
    r"""
    Calculate the surface area of a meshed region

    Parameters
    ----------
    mesh : tuple
        The tuple returned from the ``mesh_region`` function
    verts : array
        An N-by-ND array containing the coordinates of each mesh vertex
    faces : array
        An N-by-ND array indicating which elements in ``verts`` form a mesh
        element.

    Returns
    -------
    surface_area : float
        The surface area of the mesh, calculated by
        ``skimage.measure.mesh_surface_area``

    Notes
    -----
    This function simply calls ``scikit-image.measure.mesh_surface_area``, but
    it allows for the passing of the ``mesh`` tuple returned by the
    ``mesh_region`` function, entirely for convenience.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/mesh_surface_area.html>`_
    to view online example.

    """
    if mesh:
        verts = mesh.verts
        faces = mesh.faces
    else:
        if (verts is None) or (faces is None):
            raise Exception('Either mesh or verts and faces must be given')
    surface_area = skimage.measure.mesh_surface_area(verts, faces)
    return surface_area


def mesh_region(region: bool, strel=None):
    r"""
    Creates a tri-mesh of the provided region using the marching cubes
    algorithm

    Parameters
    ----------
    im : ndarray
        A boolean image with ``True`` values indicating the region of interest
    strel : ndarray
        The structuring element to use when blurring the region.  The blur is
        perfomed using a simple convolution filter.  The point is to create a
        greyscale region to allow the marching cubes algorithm some freedom
        to conform the mesh to the surface.  As the size of ``strel`` increases
        the region will become increasingly blurred and inaccurate. The default
        is a spherical element with a radius of 1.

    Returns
    -------
    mesh : tuple
        A named-tuple containing ``faces``, ``verts``, ``norm``, and ``val``
        as returned by ``scikit-image.measure.marching_cubes`` function.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/mesh_region.html>`_
    to view online example.

    """
    im = region
    if strel is None:
        if region.ndim == 3:
            strel = skimage.morphology.ball(1)
        if region.ndim == 2:
            strel = skimage.morphology.disk(1)
    pad_width = np.amax(strel.shape)
    if im.ndim == 3:
        padded_mask = np.pad(im, pad_width=pad_width, mode='constant')
        padded_mask = spim.convolve(padded_mask * 1.0,
                                    weights=strel) / np.sum(strel)
    else:
        padded_mask = np.reshape(im, (1,) + im.shape)
        padded_mask = np.pad(padded_mask, pad_width=pad_width, mode='constant')
    try:
        marching_cubes_fn = skimage.measure.marching_cubes
    except AttributeError:
        marching_cubes_fn = skimage.measure.marching_cubes_lewiner
    verts, faces, norm, val = marching_cubes_fn(padded_mask)
    result = AttrDict()
    result.verts = verts - pad_width
    result.faces = faces
    result.norm = norm
    result.val = val
    return result


def extend_slice(slices, shape, pad=1):
    r"""
    Adjust slice indices to include additional voxles around the slice.

    This function does bounds checking to ensure the indices don't extend
    outside the image.

    Parameters
    ----------
    slices : list of slice objects
         A list (or tuple) of N slice objects, where N is the number of
         dimensions in the image.
    shape : array_like
        The shape of the image into which the slice objects apply.  This is
        used to check the bounds to prevent indexing beyond the image.
    pad : int or list of ints
        The number of voxels to expand in each direction.

    Returns
    -------
    slices : list of slice objects
        A list slice of objects with the start and stop attributes respectively
        incremented and decremented by 1, without extending beyond the image
        boundaries.

    Examples
    --------
    >>> from scipy.ndimage import label, find_objects
    >>> from porespy.tools import extend_slice
    >>> im = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    >>> labels = label(im)[0]
    >>> s = find_objects(labels)

    Using the slices returned by ``find_objects``, set the first label to 3

    >>> labels[s[0]] = 3
    >>> print(labels)
    [[3 0 0]
     [3 0 0]
     [0 0 2]]

    Next extend the slice, and use it to set the values to 4

    >>> s_ext = extend_slice(s[0], shape=im.shape, pad=1)
    >>> labels[s_ext] = 4
    >>> print(labels)
    [[4 4 0]
     [4 4 0]
     [4 4 2]]

    As can be seen by the location of the 4s, the slice was extended by 1, and
    also handled the extension beyond the boundary correctly.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/extend_slice.html>`_
    to view online example.

    """
    shape = np.array(shape)
    pad = np.array(pad).astype(int)*(shape > 0)
    a = []
    for i, s in enumerate(slices):
        start = 0
        stop = shape[i]
        start = max(s.start - pad[i], 0)
        stop = min(s.stop + pad[i], shape[i])
        a.append(slice(start, stop, None))
    return tuple(a)
