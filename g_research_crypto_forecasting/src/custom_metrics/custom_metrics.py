import numpy as np
from scipy.stats import pearsonr

def weighted_correlation_coefficient(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.array([1] * len(y_true))

    weights = np.array(weights)

    return pearsonr(y_true * weights, y_pred * weights)[0]


def rmspe(y_true, y_pred, weights=None):  # TODO: fix 0 division case
    """
    Root Mean Squared Percentage Error
    """
    if weights is None:
        weights = np.array([1] * len(y_true))

    weights = np.array(weights)

    return np.sqrt(np.mean(np.square(((y_true * weights - y_pred * weights) / (y_true * weights))), axis=0)) * 100
