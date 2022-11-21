"""
Multi Otsu thresholding algorithm

from
    *A fast algorithm for multilevel thresholding*, Liao, P-S., Chen, T-S. and Chung, P-C. Journal of Information Science and Engineering. 2001 Sept;17(5):713-27.
    *A threshold selection method from gray-level histograms*, Otsu, Nobuyuki. IEEE transactions on systems, man, and cybernetics. 1979;9(1):62-6.

This code is adapted from pythreshold by BSc. Manuel Aguado Martínez.
Location of file in pythreshold: otsu_multithreshold function and its supporting functions
in pythreshold.global_th.otsu
"""

from itertools import combinations
import numpy as np

def _get_variance(hist, c_hist, cdf, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    variance = 0

    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # Cumulative histogram
        weight = c_hist[t2] - c_hist[t1 - 1]

        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]

        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0

        variance += weight * r_mean ** 2

    return variance


def _get_thresholds(hist, c_hist, cdf, classes):
    """Get the thresholds that maximize the variance between regions
    Parameters
    ----------
    hist : ndarray
        the normalized histogram of the image
    chist : ndarray
        the cumulative histogram of the image
    cdf : ndarray
        the cummulative distribution function of the histogram
    classes : int
        Number of classes to be thresholded, i.e. the number of resulting
        regions.

    Returns
    -------
    opt_thresholds : array
        Array containing the threshold values for the desired classes.
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), classes)

    max_var = 0
    opt_thresholds = None

    # Extending histograms for convenience
    c_hist = np.append(c_hist, [0])
    cdf = np.append(cdf, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing variance for the current combination of thresholds
        regions_var = _get_variance(hist, c_hist, cdf, e_thresholds)

        if regions_var > max_var:
            max_var = regions_var
            opt_thresholds = thresholds

    return opt_thresholds


def otsu_multithreshold(image=None, classes=3, nbins=256, hist=None):
    """ Runs the Otsu's multi-threshold algorithm.
    Parameters
    ----------
    image : (N, M[, ..., P]) ndarray, optional
        Grayscale input image.
    classes : int, optional
        Number of classes to be thresholded, i.e. the number of resulting
        regions.
    nbins : int, optional
        Number of bins used to calculate the histogram.
    hist : ndarray, the input image histogram

    Returns
    -------
    thresh : array
        Array containing the threshold values for the desired classes.

    Raises
    ------
    ValueError
         If ``image`` contains less grayscale value then the desired
         number of classes.
    """
    # Histogran
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(nbins))[0].astype(np.float)

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    thresh = _get_thresholds(hist, c_hist, cdf, classes)
    return thresh