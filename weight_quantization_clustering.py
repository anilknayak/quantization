import numpy as np
from sklearn.cluster import KMeans
import loss

def channel_wise_clusering_weights_quantization(weights, verbose=0):
    # Weights could have [number_of_filters, kernel_size, kernel_size]
    # Weights are in float64
    dtype_weight = np.asarray(weights).dtype
    print("Before quantization_research weights type: '{}' "
          "of memory footprint: {} bits".format(dtype_weight,
                                                loss.calculate_memory_footprint(weights)))
    if verbose:
        print("Before quantization_research weights \n  {}".format(weights))
    weights = np.asarray(weights)
    num_channels, clusters = np.shape(weights)[:2]
    print("Number of clusters: {} and number channels: {}".format(clusters,
                                                                  num_channels))
    centroids_channel_wise = []
    quantized_weights_index_channel_wise = []
    for channel in range(num_channels):
        weight = weights[channel]
        flatten_weights = weight.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(flatten_weights)
        weight_cluster_centers = sorted(kmeans.cluster_centers_)
        centroids_channel_wise.append(weight_cluster_centers)
        distances = []
        for i in range(clusters):
            distances.append(np.abs(np.subtract(weight, weight_cluster_centers[i][0])))
        quantized_weights_indexs = np.argmin(distances, axis=0)
        quantized_weights_index_channel_wise.append(quantized_weights_indexs)
    quantized_weights_index_channel_wise = np.asarray(quantized_weights_index_channel_wise,
                                                      dtype=np.int8)
    centroids_channel_wise = np.asarray(centroids_channel_wise, dtype=dtype_weight)
    memory_footprint = 8*quantized_weights_index_channel_wise.size
    memory_footprint += loss.calculate_memory_footprint(centroids_channel_wise)
    print("After quantization_research weights dtype: '{}' and weights index dtype '{}'" 
          " of total memory footprint: {} bits".format(dtype_weight,
                                           quantized_weights_index_channel_wise.dtype,
                                           memory_footprint))
    if verbose:
        print("After quantization_research weights channel wise \n {}".format(centroids_channel_wise))
        print("After quantization_research weights indexes channel wise \n {} ".format(
            quantized_weights_index_channel_wise))

    return quantized_weights_index_channel_wise, centroids_channel_wise

def construct_weights_from_quantized_weights(weights_indexes, centroids):
    constructed_weights = np.zeros(shape=weights_indexes.shape, dtype=np.float64)
    num_channels, _ = np.shape(weights_indexes)[:2]
    for channel in range(num_channels):
        weight_index = weights_indexes[channel]
        centroid = centroids[channel]
        h, w = weight_index.shape
        for j in range(w):
            for k in range(h):
                constructed_weights[channel][j][k] = centroid[weight_index[j][k]][0]
    return constructed_weights

def test_1():
    original_weights = np.asarray([
                            [
                                [2.09, -0.98, 1.48, 0.09],
                                [0.05, -0.14, -1.08, 2.12],
                                [-0.91, 1.92, 0, -1.03],
                                [1.87, 0, 1.53, 1.49]
                            ],
                            [
                                [2.09, -0.98, 1.48, 0.09],
                                [0.05, -0.14, -1.08, 2.12],
                                [-0.91, 1.92, 0, -1.03],
                                [1.87, 0, 1.53, 1.49]
                            ]
                        ], dtype=np.float64)
    weights_indexes, centroids = channel_wise_clusering_weights_quantization(original_weights, 0)
    constructed_weights = construct_weights_from_quantized_weights(weights_indexes, centroids)
    loss.MAE(original_weights, constructed_weights)
    loss.MSE(original_weights, constructed_weights)

def test_2():
    original_weights = np.asarray([[
                            [2.09, -0.98, 1.48, 0.09],
                            [0.05, -0.14, -1.08, 2.12],
                            [-0.91, 1.92, 0, -1.03],
                            [1.87, 0, 1.53, 1.49]
                        ]], dtype=np.float64)
    weights_indexes, centroids = channel_wise_clusering_weights_quantization(original_weights, 0)
    constructed_weights = construct_weights_from_quantized_weights(weights_indexes, centroids)
    loss.MAE(original_weights, constructed_weights)
    loss.MSE(original_weights, constructed_weights)

test_1()
print("="*100)
test_2()