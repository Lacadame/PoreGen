# A reduced version of porespy.metrics._funcs.py
import numpy as np
import scipy.ndimage
import scipy.fft
import psutil

from poregen.utils import AttrDict


def porosity(im):
    return (im == 1).mean()


def two_point_correlation(im, voxel_size=1, bins=100):
    r"""
    Calculate the two-point correlation function using Fourier transforms

    Parameters
    ----------
    im : ndarray
        The image of the void space on which the 2-point correlation is
        desired, in which the phase of interest is labelled as True
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so
        the user can apply the scaling to the returned results after the
        fact.
    bins : scalar or array_like
        Either an array of bin sizes to use, or the number of bins that
        should be automatically generated that span the data range. The
        maximum value of the bins, if passed as an array, cannot exceed
        the distance from the center of the image to the corner.

    Returns
    -------
    result : tpcf
        The two-point correlation function object, with named attributes:

        *distance*
            The distance between two points, equivalent to bin_centers
        *bin_centers*
            The center point of each bin. See distance
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``
        *probability_normalized*
            The probability that two points of the stated separation distance
            are within the same phase normalized to 1 at r = 0
        *probability* or *pdf*
            The probability that two points of the stated separation distance
            are within the same phase scaled to the phase fraction at r = 0

    Notes
    -----
    The fourier transform approach utilizes the fact that the
    autocorrelation function is the inverse FT of the power spectrum
    density. For background read the Scipy fftpack docs and for a good
    explanation `see this thesis
    <https://www.ucl.ac.uk/~ucapikr/projects/KamilaSuankulova_BSc_Project.pdf>`_.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/two_point_correlation.html>`_
    to view online example.

    """
    # Get the number of CPUs available to parallel process Fourier transforms
    cpus = psutil.cpu_count(logical=False)
    # Get the phase fraction of the image
    pf = porosity(im)
    if isinstance(bins, int):
        # Calculate half lengths of the image
        r_max = (np.ceil(np.min(np.shape(im))) / 2).astype(int)
        # Get the bin-size - ensures it will be at least 1
        bin_size = int(np.ceil(r_max / bins))
        # Calculate the bin divisions, equivalent to bin_edges
        bins = np.arange(0, r_max + bin_size, bin_size)
    # set the number of parallel processors to use:
    with scipy.fft.set_workers(cpus):
        # Fourier Transform and shift image
        F = scipy.fft.ifftshift(scipy.fft.rfftn(scipy.fft.fftshift(im)))
        # Compute Power Spectrum
        P = np.absolute(F**2)
        # Auto-correlation is inverse of Power Spectrum
        autoc = np.absolute(scipy.fft.ifftshift(scipy.fft.irfftn(scipy.fft.fftshift(P))))
    tpcf = _radial_profile(autoc, bins, pf=pf, voxel_size=voxel_size)
    return tpcf


def pore_size_distribution(im, bins=10, log=True, voxel_size=1):
    r"""
    Calculate a pore-size distribution based on the image produced by the
    ``porosimetry`` or ``local_thickness`` functions.

    Parameters
    ----------
    im : ndarray
        The array of containing the sizes of the largest sphere that overlaps
        each voxel.  Obtained from either ``porosimetry`` or
        ``local_thickness``.
    bins : scalar or array_like
        Either an array of bin sizes to use, or the number of bins that should
        be automatically generated that span the data range.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the retunred
        ``tuple``, since the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *R* or *logR*
            Radius, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *satn*
            Phase saturation in differential form.  For the cumulative
            saturation, just use *cfd* which is already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    Notes
    -----
    (1) To ensure the returned values represent actual sizes you can manually
    scale the input image by the voxel size first (``im *= voxel_size``)

    plt.bar(psd.R, psd.satn, width=psd.bin_widths, edgecolor='k')

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/pore_size_distribution.html>`_
    to view online example.

    """
    im = im.flatten()
    vals = im[im > 0] * voxel_size
    if log:
        vals = np.log10(vals)
    h = _parse_histogram(np.histogram(vals, bins=bins, density=True))
    cld = AttrDict()
    cld[f"{log*'Log' + 'R'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.satn = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


def _parse_histogram(h, voxel_size=1, density=True):
    delta_x = h[1]
    P = h[0]
    bin_widths = delta_x[1:] - delta_x[:-1]
    temp = P * (bin_widths)
    C = np.cumsum(temp[-1::-1])[-1::-1]
    S = P * (bin_widths)
    if not density:
        P /= np.max(P)
        temp_sum = np.sum(P * bin_widths)
        C /= temp_sum
        S /= temp_sum

    bin_edges = delta_x * voxel_size
    bin_widths = (bin_widths) * voxel_size
    bin_centers = ((delta_x[1:] + delta_x[:-1]) / 2) * voxel_size
    hist = AttrDict()
    hist.pdf = P
    hist.cdf = C
    hist.relfreq = S
    hist.bin_centers = bin_centers
    hist.bin_edges = bin_edges
    hist.bin_widths = bin_widths
    return hist


def _radial_profile(autocorr, bins, pf=None, voxel_size=1):
    r"""
    Helper functions to calculate the radial profile of the autocorrelation

    Masks the image in radial segments from the center and averages the values
    The distance values are normalized and 100 bins are used as default.

    Parameters
    ----------
    autocorr : ndarray
        The image of autocorrelation produced by FFT
    r_max : int or float
        The maximum radius in pixels to sum the image over
    bins : ndarray
        The edges of the bins to use in summing the radii, ** must be in voxels
    pf : float
        the phase fraction (porosity) of the image, used for scaling the
        normalized autocorrelation down to match the two-point correlation
        definition as given by Torquato
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : tpcf


    """
    if len(autocorr.shape) == 2:
        adj = np.reshape(autocorr.shape, [2, 1, 1])
        # use np.round otherwise with odd image sizes, the mask generated can
        # be zero, resulting in Div/0 error
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autocorr.shape) == 3:
        adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
        # use np.round otherwise with odd image sizes, the mask generated can
        # be zero, resulting in Div/0 error
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
    else:
        raise Exception('Image dimensions must be 2 or 3')
    if np.max(bins) > np.max(dt):
        msg = (
            'Bins specified distances exceeding maximum radial distance for'
            ' image size. Radial distance cannot exceed distance from center'
            ' of image to corner.'
        )
        raise Exception(msg)

    bin_size = bins[1:] - bins[:-1]
    radial_sum = _get_radial_sum(dt, bins, bin_size, autocorr)
    # Return normalized bin and radially summed autoc
    norm_autoc_radial = radial_sum / np.max(autocorr)
    h = [norm_autoc_radial, bins]
    h = _parse_histogram(h, voxel_size=1)
    tpcf = AttrDict()
    tpcf.distance = h.bin_centers * voxel_size
    tpcf.bin_centers = h.bin_centers * voxel_size
    tpcf.bin_edges = h.bin_edges * voxel_size
    tpcf.bin_widths = h.bin_widths * voxel_size
    tpcf.probability = norm_autoc_radial
    tpcf.probability_scaled = norm_autoc_radial * pf
    tpcf.pdf = h.pdf * pf
    tpcf.relfreq = h.relfreq
    return tpcf


def _get_radial_sum(dt, bins, bin_size, autocorr):
    radial_sum = np.zeros_like(bins[:-1])
    for i, r in enumerate(bins[:-1]):
        mask = (dt <= r) * (dt > (r - bin_size[i]))
        radial_sum[i] = np.sum(np.ravel(autocorr)[np.ravel(mask)], dtype=np.int64) \
            / np.sum(mask)
    return radial_sum
