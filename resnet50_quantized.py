import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import kl_divergence_quantization

# np.set_printoptions(threshold=sys.maxsize)
model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000)

# model.summary()
# for i, layer in enumerate(model._layers):
#     print(i, layer.name, layer)


def layer_weights(model, layer_no):
    layer = model._layers[layer_no]
    weights = layer.weights[0].numpy()
    weights = np.array(weights, dtype=np.float32)
    return weights

def plot_histogram(weights, bins=2048):
    flatten_matrix = weights.flatten()
    min_val = np.min(flatten_matrix)
    max_val = np.max(flatten_matrix)
    threshold = max(abs(min_val), abs(max_val))
    print("Threshold", threshold)
    print("min_val {} max_val {}".format(min_val, max_val))
    # P Distribution
    histdtl = plt.hist(flatten_matrix, bins=bins, range=(-threshold, threshold))
    hist = histdtl[0]
    hist_range = histdtl[1]
    plt.show()

weights = layer_weights(model, 10)
# plot_histogram(weights, bins=2048)

# KL divergence will fine tune the
# 1. Threshold value range of real fp32
# 2. How much bucket will give us minimum KL divergence is important
# 3. scale factor = (scale int8 / range fp32)

# kl_divergence_quantization.kl_divergence_scale(weights, num_bins=256+1)
# kl_divergence_quantization.kl_divergence_scale(weights, num_bins=512+1)
# kl_divergence_quantization.kl_divergence_scale(weights, num_bins=1024+1)
# kl_divergence_quantization.kl_divergence_scale(weights, num_bins=2048+1)
# kl_divergence_quantization.kl_divergence_scale(weights, num_bins=4096+1)
# kl_divergence_quantization.kl_divergence_scale(weights, num_bins=8192+1)
