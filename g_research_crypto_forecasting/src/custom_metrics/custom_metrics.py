import numpy as np
from scipy.stats import pearsonr


def weighted_correlation_coefficient(y_true, y_pred, weights=None):
    """
    https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
    """
    def wm(x, w):
        """Weighted Mean"""
        return np.sum(x * w) / np.sum(w)

    def wcov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - wm(x, w)) * (y - wm(y, w))) / np.sum(w)

    if weights is None:
        weights = np.array([1] * len(y_true))

    weights = np.array(weights)

    return wcov(y_true, y_pred, weights) / (
        np.sqrt(wcov(y_true, y_true, weights) * wcov(y_pred, y_pred, weights))
    )

def rmspe(y_true, y_pred, weights=None):  # TODO: fix 0 division case
    """
    Root Mean Squared Percentage Error
    """
    if weights is None:
        weights = np.array([1] * len(y_true))

    weights = np.array(weights)

    return np.sqrt(np.mean(np.square(((y_true * weights - y_pred * weights) / (y_true * weights))), axis=0)) * 100

# small tests
if __name__ == "__main__":
    y_true = [1, 2, 2, 3, 4]
    y_pred = [1, 1.1, 1, 5, 6]
    weights = [1, 1, 1, 0, 0]
    print(f"""
    No weights:
        {weighted_correlation_coefficient(y_true, y_pred)}
        
    Weights:
        {weighted_correlation_coefficient(y_true, y_pred, weights)}
    """)