import torch
import torch.nn as nn

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        m.reset_parameters()

def optimizer_to(optimizer, device):
    """
    Move the optimizer tensors to device before training.

    Solves restore issue:
    https://github.com/atomistic-machine-learning/schnetpack/issues/126
    https://github.com/pytorch/pytorch/issues/2830
    """
   
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


import statistics
import numpy as np

def mean_squared_error(data):
    """ Calculate the mean squared error between data_groups

    args:
        data (list) -- (2, n) list where n is the number of points to be compared

    returns:
        mse
    """

    return statistics.mean((np.array(data[0]) - np.array(data[1]))**2)

def root_mean_square_error(data):
    """ Calculate the root mean squared error between data_groups

    args:
        data (list) -- (2, n) list where n is the number of points to be compared

    returns:
        rmse
    """

    return np.sqrt(mean_squared_error(data))

def mean_absolute_error(data):
    """ Calculate the mean absolute error between data_groups

    args:
        data (list) -- (2, n) list where n is the number of points to be compared

    returns:
        mae
    """

    return statistics.mean(abs(np.array(data[0]) - np.array(data[1])))

def correlation(data):
    """ Calculate the correlation error between data_groups

        args:
            data (list) -- (2, n) list where n is the number of points to be compared
        
        returns:
            r^2 value
    """

    numerator = statistics.mean((np.array(data[0]) - statistics.mean(data[0]))*
                                 (np.array(data[1]) - statistics.mean(data[1])))
    denominator = statistics.stdev(data[0]) * statistics.stdev(data[1])

    return numerator / denominator


def combine_mean_var_from_subsets(means: np.ndarray, variances: np.ndarray, num_obs: np.ndarray, ddof=0):

    if not (means.shape == variances.shape and means.shape == num_obs.shape):
        raise ValueError(f"All input arrays are expected to have the same size, got: "
                            f"`means.shape`={means.shape}, `variances.shape`={variances.shape}, "
                            f"`num_obs.shape`={num_obs.shape}")

    total_num_obs = num_obs.sum()
    mean = np.sum(means*num_obs)/total_num_obs
    if ddof == 0:
        var = np.sum((num_obs*(variances+(means-mean)**2)))/(total_num_obs)
    else:
        num_obs_ddof = num_obs-ddof
        var = np.sum((num_obs*(num_obs_ddof/num_obs*variances+(means-mean)**2)))/(total_num_obs-ddof)
    return mean, var