import numpy as np
from typing import Tuple
from sklearn.preprocessing import label_binarize

def calibration_curve(y_true: np.array, y_prob: np.array, *, normalize: bool=False, n_bins: int=5, strategy: str="uniform") -> Tuple[np.array, np.array, np.array]:
    """
    Compute the calibration curve for binary classification.

    This function calculates the calibration curve for binary classification problems.
    It quantifies the relationship between the predicted probabilities (`y_prob`) and the actual outcomes
    (`y_true`) for different probability bins.

    Note that this function is an adjustment of sklearns calibration curve (sklearn.calibration.calibration_curve).
    Here, it reacts to potential overlapping bounds of bins, which then would lead to an error.

    Parameters:
        y_true (np.array): Ground truth binary labels (0 or 1).

        y_prob (np.array): Predicted probabilities for positive class (class 1).

        normalize (bool, optional): Whether to normalize the predicted probabilities into the interval [0, 1].
            Default is False.

        n_bins (int, optional): Number of bins to divide the predicted probabilities into.
            Default is 5.

        strategy (str, optional): The strategy for bin creation:
            - "uniform": Bins are equally spaced between 0 and 1.
            - "quantile": Bins are determined by quantiles of the predicted probabilities.
            Default is "uniform".

    Returns:
        Tuple[np.array, np.array, np.array]: A tuple containing three arrays:
            - prob_true (np.array): The observed fractions of positive outcomes in each bin.
            - prob_pred (np.array): The mean predicted probabilities for each bin.
            - weights (np.array): The weights associated with each bin.

    Raises:
        ValueError: If the provided labels in `y_true` are not binary (more than two unique values).
        ValueError: If `strategy` is not "uniform" or "quantile".
        ValueError: If `normalize` is False and `y_prob` contains values outside the [0, 1] range.
    """

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is " "set to False.")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. " "Provided labels %s." % labels)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8

    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy " "must be either 'quantile' or 'uniform'.")

    # Set bin bounds
    while np.mean([int(bins[i] <= bins[i + 1]) for i in range(len(bins) - 1)]) < 1:
        for i in range(len(bins) - 1):
            if bins[i] > bins[i + 1]:
                bins[i] = bins[i + 1]

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    weights = bin_total[nonzero] / sum(bin_total)

    return prob_true, prob_pred, weights


def ece(fraction_of_positives: np.array, mean_predicted_value: np.array, weights: np.array) -> float:
    """
    Calculate the Expected Calibration Error (ECE).

    Args:
        fraction_of_positives (np.array): An array representing the observed fraction of positive outcomes.
        mean_predicted_value (np.array): An array representing the mean predicted probability for each bin or group.
        weights (np.array): An array representing the weights associated with each bin or group.

    Returns:
        float: The Expected Calibration Error (ECE) as a percentage.
        
    Raises:
        AssertionError: If the lengths of `fraction_of_positives`, `mean_predicted_value`, and `weights` do not match.
    """

    N = len(fraction_of_positives)  # can differ from b_bins for uniform strategy
    assert N == len(mean_predicted_value) == len(weights)

    ece_ = sum([weights[i] * abs(fraction_of_positives[i] - mean_predicted_value[i]) for i in range(N)])
    return ece_ * 100



def mce(fraction_of_positives: np.array, mean_predicted_value: np.array) -> float:
    """
    Calculate the Maximum Calibration Error (MCE).

    The Maximum Calibration Error represents the highest discrepancy between observed
    fraction of positive outcomes and mean predicted probabilities over all bins or groups.

    Args:
        fraction_of_positives (np.array): An array representing the observed fraction of positive outcomes.
        mean_predicted_value (np.array): An array representing the mean predicted probability for each bin or group.

    Returns:
        float: The Maximum Calibration Error (MCE) as a percentage. It quantifies the worst-case
        calibration error, indicating how far the model's predicted probabilities are from the actual outcomes.

    Raises:
        AssertionError: If the lengths of `fraction_of_positives` and `mean_predicted_value` do not match.
    """

    N = len(fraction_of_positives)
    assert N == len(mean_predicted_value)
    errors = [abs(fraction_of_positives[i] - mean_predicted_value[i]) for i in range(N)]

    return max(errors) * 100


def mce_rel(fraction_of_positives: np.array, mean_predicted_value: np.array) -> float:
    """
    Calculate the Maximum Relative Calibration Error (rel MCE).

    The Maximum Relative Calibration Error measures the highest relative discrepancy between the observed
    fraction of positive outcomes and mean predicted probabilities over all bins or groups.

    Args:
        fraction_of_positives (np.array): An array representing the observed fraction of positive outcomes.
        mean_predicted_value (np.array): An array representing the mean predicted probability for each bin or group.

    Returns:
        float: The Maximum Relative Calibration Error (rel MCE). It quantifies the worst-case calibration
        error relative to the predicted probabilities, indicating how far the model's predictions deviate
        from the actual outcomes in a relative sense.

    Raises:
        AssertionError: If the lengths of `fraction_of_positives` and `mean_predicted_value` do not match.
    """

    N = len(fraction_of_positives)
    assert N == len(mean_predicted_value)
    rel_dists = [abs(fraction_of_positives[i] - mean_predicted_value[i]) / abs(mean_predicted_value[i]) for i in range(N)]

    return max(rel_dists)
