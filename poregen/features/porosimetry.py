"""
Adapted from porespy.filters.porosimetry, but with less boilerplate
"""

import numpy as np
# import porespy
import skimage.morphology
import scipy.ndimage
import edt

from . import fftmorphology


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
        strel = ps_disk
        strel_2 = skimage.morphology.disk
    else:
        strel = ps_ball
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
            imtemp = fftmorphology.fftmorphology(imtemp, strel(r),
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
    return borders(shape=shape,
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


def ps_disk(r, smooth=True):
    r"""
    Creates circular disk structuring element for morphological operations

    Parameters
    ----------
    r : float or int
        The desired radius of the structuring element
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    disk : ndarray
        A 2D numpy bool array of the structring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_disk.html>`_
    to view online example.

    """
    disk = ps_round(r=r, ndim=2, smooth=smooth)
    return disk


def ps_ball(r, smooth=True):
    r"""
    Creates spherical ball structuring element for morphological operations

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    ball : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_ball.html>`_
    to view online example.

    """
    ball = ps_round(r=r, ndim=3, smooth=smooth)
    return ball


def ps_round(r, ndim, smooth=True):
    r"""
    Creates round structuring element with the given radius and dimensionality

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    strel : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_round.html>`_
    to view online example.

    """
    rad = int(np.ceil(r))
    other = np.ones([2*rad + 1 for i in range(ndim)], dtype=bool)
    other[tuple(rad for i in range(ndim))] = False
    if smooth:
        ball = edt.edt(other) < r
    else:
        ball = edt.edt(other) <= r
    return ball


def ps_rect(w, ndim):
    r"""
    Creates rectilinear structuring element with the given size and
    dimensionality

    Parameters
    ----------
    w : scalar
        The desired width of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.

    Returns
    -------
    strel : D-aNrray
        A numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_rect.html>`_
    to view online example.

    """
    if ndim == 2:
        from skimage.morphology import square
        strel = square(w)
    if ndim == 3:
        from skimage.morphology import cube
        strel = cube(w)
    return strel


def faces(shape, inlet: int = None, outlet: int = None):
    r"""
    Generate an image with ``True`` values on the specified ``inlet`` and
    ``outlet`` faces

    Parameters
    ----------
    shape : list
        The ``[x, y, z (optional)]`` shape to generate. This will likely
        be obtained from ``im.shape`` where ``im`` is the image for which
        an array of faces is required.
    inlet : int
        The axis where the faces should be added (e.g. ``inlet=0`` will
        put ``True`` values on the ``x=0`` face). A value of ``None``
        bypasses the addition of inlets.
    outlet : int
        Same as ``inlet`` except for the outlet face. This is optional. It
        can be be applied at the same time as ``inlet``, instead of
        ``inlet`` (if ``inlet`` is set to ``None``), or ignored
        (if ``outlet = None``).

    Returns
    -------
    faces : ndarray
        A boolean image of the given ``shape`` with ``True`` values on the
        specified ``inlet`` and/or ``outlet`` face(s).

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/faces.html>`_
    to view online example.

    """
    im = np.zeros(shape, dtype=bool)
    # Parse inlet and outlet
    if inlet is not None:
        im = np.swapaxes(im, 0, inlet)
        im[0, ...] = True
        im = np.swapaxes(im, 0, inlet)
    if outlet is not None:
        im = np.swapaxes(im, 0, outlet)
        im[-1, ...] = True
        im = np.swapaxes(im, 0, outlet)
    if (inlet is None) and (outlet is None):
        raise Exception('Both inlet and outlet were given as None')
    return im


def borders(
    shape,
    thickness: int = 1,
    mode: str = 'edges'
):
    r"""
    Creates an array of specified size with corners, edges or faces
    labelled as ``True``.

    This can be used as mask to manipulate values laying on the perimeter
    of an image.

    Parameters
    ----------
    shape : array_like
        The shape of the array to return.  Can be either 2D or 3D.
    thickness : scalar (default is 1)
        The number of pixels/voxels layers to place along perimeter.
    mode : string
        The type of border to create.  Options are 'faces', 'edges'
        (default) and 'corners'.  In 2D 'corners' and 'edges' give the
        same result.

    Returns
    -------
    image : ndarray
        An ndarray of specified shape with ``True`` values at the
        perimeter and ``False`` elsewhere

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/borders.html>`_
    to view online example.

    """
    ndims = len(shape)
    t = thickness
    border = np.ones(shape, dtype=bool)
    if mode == 'faces':
        if ndims == 2:
            border[t:-t, t:-t] = False
        if ndims == 3:
            border[t:-t, t:-t, t:-t] = False
    elif mode == 'edges':
        if ndims == 2:
            border[t:-t, 0::] = False
            border[0::, t:-t] = False
        if ndims == 3:
            border[0::, t:-t, t:-t] = False
            border[t:-t, 0::, t:-t] = False
            border[t:-t, t:-t, 0::] = False
    elif mode == 'corners':
        if ndims == 2:
            border[t:-t, 0::] = False
            border[0::, t:-t] = False
        if ndims == 3:
            border[t:-t, 0::, 0::] = False
            border[0::, t:-t, 0::] = False
            border[0::, 0::, t:-t] = False
    return border
