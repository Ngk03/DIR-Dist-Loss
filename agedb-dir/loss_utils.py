import numpy as np
from scipy.stats import gaussian_kde
from typing import Union, Callable

def get_label_distribution(labels: np.ndarray, 
                           bw_method: Union[str, float, Callable[[gaussian_kde], float], None] = 0.5, 
                           min_label: Union[int, float] = 21, 
                           max_label: Union[int, float] = 100, 
                           step: float = 1.0) -> np.ndarray:
    """
    Get the label distribution using kernel density estimation.

    Parameters:
        labels (np.ndarray): The label array to be operated.
        bw_method (Union[str, float, Callable[[gaussian_kde], float], None]): 
            The method used to calculate the estimator bandwidth. This can be 'scott', 'silverman', a scalar constant or a callable.
            If a scalar, this will be used directly as kde.factor. If a callable, it should take a gaussian_kde instance as only parameter and return a scalar.
            If None (default), nothing happens; the current kde.covariance_factor method is kept. 
            See details at https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.set_bandwidth.html.
        min_label (Union[int, float]): The theoretical minimum of the label. Values below this will be assigned a zero probability.
        max_label (Union[int, float]): The theoretical maximum of the label. Values above this will be assigned a zero probability.
        step (float): The interval between discrete labels in the estimated distribution.

    Returns:
        np.ndarray: The estimated density for the labels.

    Note:
        `min_label` and `max_label` represent the theoretical range of possible labels. 
        Any label values below `min_label` or above `max_label` will have an assigned probability of zero in the output distribution.
        `step` defines the resolution of the label distribution.
    """
    # Ensure the labels are within the theoretical range
    labels = labels[(labels >= min_label) & (labels <= max_label)]

    # Perform kernel density estimation
    kde = gaussian_kde(labels, bw_method=bw_method)
    
    # Create an array of possible label values within the theoretical range with the specified step
    x = np.arange(min_label, max_label + step, step)
    
    # Estimate the density for these label values
    density_estimation = kde(x)
    density_estimation /= density_estimation.sum()

    return density_estimation

def get_batch_label_distribution(density: np.ndarray, batch_size: int, region_adjustment: float = 0.5) -> np.ndarray:
    """
    Acquire the label distribution of one batch based on kernel density estimation.

    Parameters:
        density (np.ndarray): Estimated density values for each label.
        batch_size (int): Number of samples in the batch.
        region_adjustment (float): Adjustment factor for distributing residuals. Default is 0.5.

    Returns:
        np.ndarray: Label distribution for the batch, summing up to 'batch_size'.

    Explanation:
        This function calculates the label distribution for a batch based on the estimated density values.
        It scales the density values with 'batch_size' to determine the number of samples for each label.
        The distribution ensures that the total sum matches 'batch_size'.
        Residual differences due to rounding are adjusted to meet 'batch_size', using 'region_adjustment' to control the range.
    """
    num_density = density * batch_size
    range_res = int(region_adjustment * len(density))
    batch_label_distribution = np.zeros_like(num_density)

    forward_cumsum = num_density.cumsum()
    backward_cumsum = num_density[::-1].cumsum()[::-1]
    forward_index = np.searchsorted(forward_cumsum, 1)
    backward_index = len(backward_cumsum) - np.searchsorted(backward_cumsum[::-1], 1) - 1

    forward_index_cumsum = round(forward_cumsum[forward_index])
    backward_index_cumsum = round(backward_cumsum[backward_index])

    batch_label_distribution[forward_index] = forward_index_cumsum
    batch_label_distribution[forward_index + 1:backward_index] = np.round(num_density[forward_index + 1:backward_index])
    batch_label_distribution[backward_index] = backward_index_cumsum
    
    res_sum = batch_size - int(batch_label_distribution.sum())
    maximum_index = batch_label_distribution.argmax()

    if abs(res_sum) <= range_res:
        left_index = maximum_index - abs(res_sum) // 2
        right_index = left_index + abs(res_sum)
        batch_label_distribution[left_index:right_index] += np.sign(res_sum)
    else:
        iters = res_sum // range_res
        remainder = res_sum % range_res
        left_index = maximum_index - range_res // 2
        right_index = left_index + range_res
        batch_label_distribution[left_index:right_index] += iters * np.sign(res_sum)
        batch_label_distribution[maximum_index] += remainder

    return batch_label_distribution

def get_batch_theoretical_labels(density: np.ndarray, batch_size: int, min_label: int, step: float = 0.5) -> np.ndarray:
    """
    Generate theoretical labels for a batch based on a theoretical label distribution estimated by kernel density estimation.

    Parameters:
        density (np.ndarray): The estimated density values for each label obtained from kernel density estimation.
                              This 1D array represents the density values for each possible label.
        batch_size (int): The number of samples in the batch to generate theoretical labels for.
        min_label (int): The minimum label value to start assigning from.
        step (float): The interval between discrete labels in the estimated distribution.

    Returns:
        np.ndarray: An array of theoretical labels following the batch label distribution.
    """
    batch_label_distribution = get_batch_label_distribution(density, batch_size)
    cumulative_distribution = np.cumsum(batch_label_distribution).astype(int)

    batch_theoretical_labels = np.zeros(batch_size)
    current_label = min_label
    num_labels = len(density)

    for i in range(num_labels):
        start_index = 0 if i == 0 else cumulative_distribution[i - 1]
        end_index = cumulative_distribution[i]
        batch_theoretical_labels[start_index:end_index] = current_label
        current_label += step

    return batch_theoretical_labels