"""
Adapted from porespy.filters.porosimetry, but with a more efficient 
implementation, for using in a torch dataloader pipeline.
"""

import numpy as np
import porespy
import skimage.morphology
import scipy.ndimage
import edt


def local_thickness(im, sizes=25, divs=1):
    return porosimetry(im, sizes=sizes, divs=divs, access_limited=False)


def porosimetry(
    im,
    sizes: int = 25,
    inlets=None,
    access_limited: bool = True,
    divs=1,
):
    r"""
    Performs a porosimetry simulution on an image.

    Parameters
    ----------
    im : ndarray
        An ND image of the porous material containing ``True`` values in the
        pore space.
    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are
        used directly.  If a scalar is provided then that number of points
        spanning the min and max of the distance transform are used.
    inlets : ndarray, boolean
        A boolean mask with ``True`` values indicating where the invasion
        enters the image.  By default all faces are considered inlets,
        akin to a mercury porosimetry experiment.  Users can also apply
        solid boundaries to their image externally before passing it in,
        allowing for complex inlets like circular openings, etc.
        This argument is only used if ``access_limited`` is ``True``.
    access_limited : bool
        This flag indicates if the intrusion should only occur from the
        surfaces (``access_limited`` is ``True``, which is the default),
        or if the invading phase should be allowed to appear in the core
        of the image.  The former simulates experimental tools like
        mercury intrusion porosimetry, while the latter is useful for
        comparison to gauge the extent of shielding effects in the sample.

    divs : int or array_like
        The number of times to divide the image for parallel processing.
        If ``1`` then parallel processing does not occur.  ``2`` is
        equivalent to ``[2, 2, 2]`` for a 3D image.  The number of cores
        used is specified in ``porespy.settings.ncores`` and defaults to
        all cores.

    Returns
    -------
    image : ndarray
        A copy of ``im`` with voxel values indicating the sphere radius at
        which it becomes accessible from the ``inlets``.  This image can be
        used to find invading fluid configurations as a function of
        applied capillary pressure by applying a boolean comparison:
        ``inv_phase = im > r`` where ``r`` is the radius (in voxels) of
        the invading sphere.  Of course, ``r`` can be converted to
        capillary pressure using a preferred model.

    Notes
    -----
    There are many ways to perform this filter, and PoreSpy offers 3,
    which users can choose between via the ``mode`` argument. These
    methods all work in a similar way by finding which foreground voxels
    can accomodate a sphere of a given radius, then repeating for smaller
    radii.

    See Also
    --------
    local_thickness

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/porosimetry.html>`_
    to view online example.

    """
    im = np.squeeze(im)
    dt = edt.edt(im > 0)

    if inlets is None:
        inlets = get_border(im.shape, mode="faces")

    if isinstance(sizes, int):
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]

    if im.ndim == 2:
        strel = porespy.tools.ps_disk
        strel_2 = skimage.morphology.disk
    else:
        strel = porespy.tools.ps_ball
        strel_2 = skimage.morphology.ball

    if isinstance(divs, int):
        divs = [divs]*im.ndim

    imresults = np.zeros(np.shape(im))
    for r in sizes:
        imtemp = dt >= r
        if access_limited:
            imtemp = trim_disconnected_blobs(imtemp, inlets,
                                             strel=strel_2(1))
        if np.any(imtemp):
            imtemp = porespy.filters.fftmorphology(imtemp, strel(r),
                                                   mode="dilation")
            imresults[(imresults == 0) * imtemp] = r
    return imresults


def get_border(shape, thickness=1, mode='edges'):
    r"""
    Create an array with corners, edges or faces labelled as ``True``.

    This can be used as mask to manipulate values laying on the perimeter of
    an image.

    Parameters
    ----------
    shape : array_like
        The shape of the array to return.  Can be either 2D or 3D.
    thickness : scalar (default is 1)
        The number of pixels/voxels to place along perimeter.
    mode : string
        The type of border to create.  Options are 'faces', 'edges' (default)
        and 'corners'.  In 2D 'faces' and 'edges' give the same result.

    Returns
    -------
    image : ndarray
        An ndarray of specified shape with ``True`` values at the perimeter
        and ``False`` elsewhere.

    Notes
    -----
    The indices of the ``True`` values can be found using ``numpy.where``.

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> mask = ps.tools.get_border(shape=[3, 3], mode='corners')
    >>> print(mask)
    [[ True False  True]
     [False False False]
     [ True False  True]]
    >>> mask = ps.tools.get_border(shape=[3, 3], mode='faces')
    >>> print(mask)
    [[ True  True  True]
     [ True False  True]
     [ True  True  True]]

    `Click here
    <https://porespy.org/examples/tools/reference/get_border.html>`_
    to view online example.

    """
    return porespy.generators.borders(shape=shape,
                                      thickness=thickness,
                                      mode=mode)


def trim_disconnected_blobs(im, inlets, strel=None):
    r"""
    Removes foreground voxels not connected to specified inlets.

    Parameters
    ----------
    im : ndarray
        The image containing the blobs to be trimmed
    inlets : ndarray or tuple of indices
        The locations of the inlets.  Can either be a boolean mask the
        same shape as ``im``, or a tuple of indices such as that returned
        by the ``where`` function.  Any voxels *not* connected directly to
        the inlets will be trimmed.
    strel : array-like
        The neighborhood over which connectivity should be checked. It
        must be symmetric and the same dimensionality as the image. It is
        passed directly to the ``scipy.ndimage.label`` function as the
        ``structure`` argument so refer to that docstring for additional
        info.

    Returns
    -------
    image : ndarray
        An array of the same shape as ``im``, but with all foreground
        voxels not connected to the ``inlets`` removed.

    See Also
    --------
    find_disconnected_voxels, find_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_disconnected_blobs.html>`_
    to view online example.

    """
    if isinstance(inlets, tuple):
        temp = np.copy(inlets)
        inlets = np.zeros_like(im, dtype=bool)
        inlets[temp] = True
    elif (inlets.shape == im.shape) and (inlets.max() == 1):
        inlets = inlets.astype(bool)
    else:
        raise Exception("inlets not valid, refer to docstring for info")
    if strel is None:
        if im.ndim == 3:
            strel = skimage.morphology.cube(3)
        else:
            strel = skimage.morphology.square(3)
    labels = scipy.ndimage.label(inlets + (im > 0), structure=strel)[0]
    keep = np.unique(labels[inlets])
    keep = keep[keep > 0]
    im2 = np.isin(labels, keep)
    im2 = im2 * im
    return im2
