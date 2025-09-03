import numpy as np

import skimage.morphology
import scipy.ndimage
from skimage.segmentation import clear_border


def fill_blind_pores(im, conn: int = None, surface: bool = False):
    r"""
    Fills all blind pores that are isolated from the main void space.

    Parameters
    ----------
    im : ndarray
        The image of the porous material

    Returns
    -------
    im : ndarray
        A version of ``im`` but with all the disconnected pores removed.
    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors,
        while for the 3D the options are 6 and 26, similarily for square
        and diagonal neighbors. The default is the maximum option.
    surface : bool
        If ``True``, any isolated pore regions that are connected to the
        sufaces of the image are but not connected to the main percolating
        path are also removed. When this is enabled, only the voxels
        belonging to the largest region are kept. This can be
        problematic if image contains non-intersecting tube-like structures,
        for instance, since only the largest tube will be preserved.

    See Also
    --------
    find_disconnected_voxels
    trim_nonpercolating_paths

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/fill_blind_pores.html>`_
    to view online example.

    """
    im = np.copy(im)
    holes = find_disconnected_voxels(im, conn=conn, surface=surface)
    im[holes] = False
    return im


def find_disconnected_voxels(im, conn: int = None, surface: bool = False):
    r"""
    Identifies all voxels that are not connected to the edge of the image.

    Parameters
    ----------
    im : ndarray
        A Boolean image, with ``True`` values indicating the phase for which
        disconnected voxels are sought.
    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors,
        while for the 3D the options are 6 and 26, similarily for square
        and diagonal neighbors. The default is the maximum option.
    surface : bool
        If ``True`` any isolated regions touching the edge of the image are
        considered disconnected.

    Returns
    -------
    image : ndarray
        An ndarray the same size as ``im``, with ``True`` values indicating
        voxels of the phase of interest (i.e. ``True`` values in the original
        image) that are not connected to the outer edges.

    See Also
    --------
    fill_blind_pores, trim_floating_solid

    Notes
    -----
    This function is just a convenient wrapper around the ``clear_border``
    function of ``scikit-image``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_disconnected_voxels.html>`_
    to view online example.

    """

    if im.ndim == 2:
        if conn == 4:
            strel = skimage.morphology.disk(1)
        elif conn in [None, 8]:
            strel = skimage.morphology.square(3)
        else:
            raise Exception("Received conn is not valid")
    elif im.ndim == 3:
        if conn == 6:
            strel = skimage.morphology.ball(1)
        elif conn in [None, 26]:
            strel = skimage.morphology.cube(3)
        else:
            raise Exception("Received conn is not valid")
    labels, N = scipy.ndimage.label(input=im, structure=strel)
    if not surface:
        holes = clear_border(labels=labels) > 0
    else:
        keep = set(np.unique(labels))
        for ax in range(labels.ndim):
            labels = np.swapaxes(labels, 0, ax)
            keep.intersection_update(set(np.unique(labels[0, ...])))
            keep.intersection_update(set(np.unique(labels[-1, ...])))
            labels = np.swapaxes(labels, 0, ax)
        holes = np.isin(labels, list(keep), invert=True)
    return holes
