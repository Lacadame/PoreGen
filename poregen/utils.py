import numpy as np
import scipy.stats


class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_minibatch_sizes(n: int,
                        b: int
                        ) -> list[int]:
    if n % b == 0:
        return [b]*(n//b)
    else:
        return [b]*(n//b) + [n % b]


def inverse_cdf_histogram(z):
    histogram, bin_edges = np.histogram(z, bins='auto', density=True)
    hist_dist = scipy.stats.rv_histogram((histogram, bin_edges))

    return hist_dist.ppf  # ppf is the inverse of the CDF
