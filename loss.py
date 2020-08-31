import numpy as np
from math import log2

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

# calculate the kl divergence
def kl_divergence(p, q, esp=0.0001):
    p += esp
    q += esp
    dkl = np.sum(p[i] * np.log2(p[i] / q[i]) for i in range(len(p)))
    print("KL divergence loss due to quantization {}".format(sum(dkl)))
    return dkl

# calculate the js divergence
def js_divergence(p, q, esp=0.0001):
    p += esp
    q += esp
    m = 0.5 * (p + q)
    djs = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    print("JS divergence loss due to quantization {}".format(sum(djs)))
    return djs

def calculate_memory_footprint(data):
    dtype_weight = np.asarray(data).dtype
    finfo = np.finfo(dtype_weight)
    size = finfo.bits * data.size
    return size
