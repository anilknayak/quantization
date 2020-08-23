import numpy as np


def MAE(original_weights, constructed_weights):
    difference = original_weights - constructed_weights
    loss = np.sum(np.abs(difference)) / original_weights.size
    print("MAE loss due to quantization_research {:0.5f}".format(loss))
    return loss


def MSE(original_weights, constructed_weights):
    difference = original_weights - constructed_weights
    loss = np.sum(np.power(difference, 2)) / original_weights.size
    print("MSE loss due to quantization_research {:0.5f}".format(loss))
    return loss


def calculate_memory_footprint(data):
    dtype_weight = np.asarray(data).dtype
    finfo = np.finfo(dtype_weight)
    size = finfo.bits * data.size
    return size
