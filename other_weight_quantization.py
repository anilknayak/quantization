import numpy as np

def quantize_weights(weights, value=1, opt_type="min_max", axis=0):
    # min_max, threshold, probability, -1&1, -1&0&1, l1norm
    # if threshold is None then not threshold is used for binary quantization_research
    # binary_connect_convolution and binary_connect_affine: -1 and 1
    # nnabla implementation, binary_connect_xxxx: -1, 0 and 1
    # hard sigmoid (probability): max(0, min(1, x+1/2))
    # binary_weight_convolution or binary_weight_affine: l1 norm

    new_weights = np.zeros(weights.shape)
    if "-1&1" == opt_type:
        new_weights[weights >= 0] = 1
        new_weights[weights < 0] = -1
    elif opt_type == "-1&0&1":
        new_weights[weights > 0] = 1
        new_weights[weights == 0] = 0
        new_weights[weights < 0] = -1
    elif opt_type == "probability":
        weights = np.maximum(0, np.minimum(1, (weights + 1) / 2))
        new_weights[weights >= value] = 1
        new_weights[weights < value] = -1
    elif opt_type == "threshold":
        new_weights[weights >= value] = 1
        new_weights[weights < value] = -1
    elif opt_type == "min_max":
        min_val = np.amin(weights, axis=axis)  # 0-col
        max_val = np.amax(weights, axis=axis)  # 0-col
        avg_val = (np.average(min_val) + np.average(max_val)) / 2
        new_weights[weights >= avg_val] = 1
        new_weights[weights < avg_val] = -1
    # elif opt_type == "l1norm":
    #     norm = np.sum(np.abs(weights)) / np.prod(weights.shape)
    #     new_weights[weights >= value] = 1
    #     new_weights[weights < 1-value] = -1
    new_weights = new_weights.astype(np.int8)
    return new_weights

old_weights = np.asarray([
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
new_weights = quantize_weights(old_weights, value=1, opt_type="min_max", axis=0)
print(new_weights)
new_weights = quantize_weights(old_weights, value=0.5, opt_type="threshold")
print(new_weights)
new_weights = quantize_weights(old_weights, opt_type="-1&1")
print(new_weights)
new_weights = quantize_weights(old_weights, opt_type="-1&0&1")
print(new_weights)