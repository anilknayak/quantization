import numpy as np
from scipy import stats
import sys

# np.set_printoptions(threshold=sys.maxsize)

def smooth_distribution(matrix, eps=0.0001):
    is_zeros = (matrix == 0).astype(np.float32)
    is_nonzeros = (matrix != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = matrix.size - n_zeros
    eps1 = 0
    if n_nonzeros != 0:
        eps1 = eps * float(n_zeros) / float(n_nonzeros)
    hist = matrix.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    return hist

# http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
# http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
# The reference distribution is `q`, and the candidate distribution is `p`.
#     `q` is a truncated version of the original distribution.

# kl_divergence Algo
# 1. P (sample distribution) [a=3/5, b=1/5, c=1/5]
# 2. Q (sample distibution) [a=5/9, b=3/9, d=1/9]
# 3. define eps=0.0001 for smoothing
#           (However, in practice, two distributions P and Q are derived
#            from observations and sample counting, that is, from frequency distributions. It
#           is unreasonable to predict in the derived probability distribution that an event is
#           completely impossible since we must take into account the possibility of unseen
#           events. A smoothing method can be used to derive the probability distribution
#           from an observed frequency distribution
# 4. SP = (a, b, c) , SQ = (a, b, d)  sample distribution observed
# 5. SU = (SP union SQ) = (a, b, c, d)
# 6. By smoothing missing symbols are added to each distribution P and Q with a small probability ϵ
#       d get added to P and c get added to Q
# 7. Psmoothed = (a : 3/5 − ϵ/3, b : 1/5 − ϵ/3, c : 1/5 − ϵ/3, d : ϵ)
# 8. Qsmoothed = (a : 5/9 − ϵ/3, b : 3/9 − ϵ/3, c : ϵ, d : 1/9 − ϵ/3).
# 9. Then KLdivergence(Psmoothed,Qsmoothed) will be computed

def kl_divergence_scale(weights, quantized_dtype='int8', num_quantized_bins=255, num_bins=8001, eps=0.0001):
    # print("KL Divergence quantized_dtype {} num_quantized_bins {} num_bins {}".format(
    #     quantized_dtype, num_quantized_bins, num_bins))
    flatten_weights = weights.flatten()
    assert isinstance(flatten_weights, np.ndarray)
    min_val = np.min(flatten_weights)
    max_val = np.max(flatten_weights)
    print("Real value float32 min_val {} , max_val {}".format(min_val,max_val))
    # print("Min {} Max {}".format(min_val, max_val))
    threshold = max(abs(min_val), abs(max_val))
    # print("Threshold {}".format(threshold))
    if min_val >= 0 and quantized_dtype in ['uint8']:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1
    # print("num_quantized_bins {}".format(num_quantized_bins))
    hist, hist_edges = np.histogram(flatten_weights, bins=num_bins, range=(-threshold, threshold))
    # print("hist {}, size {}".format(hist, np.shape(hist)))
    # print("hist_edges {}, size {}".format(hist_edges, np.shape(hist_edges)))
    zero_bin_idx = num_bins // 2
    # print("zero_bin_idx {}".format(zero_bin_idx))
    num_half_quantized_bins = num_quantized_bins // 2
    # print("num_half_quantized_bins {}".format(num_half_quantized_bins))
    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    # print("thresholds {}".format(thresholds))
    divergence = np.zeros_like(thresholds)
    # print("divergence {}".format(divergence))
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # print("quantized_bins {}".format(quantized_bins))
    # p is the whole range of values in the unquantized buckets
    # q is the quantized buckets such that part of p is merged and put to one index of q
    # print("="*40)
    for i in range(num_quantized_bins // 2, num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        # print("p_bin_idx_start {} p_bin_idx_stop {}".format(p_bin_idx_start, p_bin_idx_stop))
        thresholds[i - num_half_quantized_bins] = max(abs(hist_edges[p_bin_idx_start]),abs(hist_edges[p_bin_idx_stop]))
        # print("thresholds {} ".format(thresholds))
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]
        # sliced_nd_hist recursively take the histogram values
        # next iteration will consider the values of previous iteration
        # is like expanding from the center
        # print("sliced_nd_hist {} ".format(sliced_nd_hist))
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # print("p {} ".format(p))
        is_nonzeros = (p != 0).astype(np.int32)
        # print("is_nonzeros {} ".format(is_nonzeros))
        # calculate how many bins should be merged to generate quantized distribution q
        # sliced_nd_hist.size : part of p
        # num_quantized_bins : size of q
        # num_merged_bins : index of q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins
        # print("num_merged_bins {} ".format(num_merged_bins))
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            # print("Merged = start {} stop {}".format(start, stop))
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        # print("quantized_bins merged {}".format(quantized_bins))
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # print("quantized_bins merged outlier at last {}".format(quantized_bins))
        # expand quantized_bins into p.size bins
        q = np.zeros(p.size, dtype=np.float32)
        # print("q {}".format(q))
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            # print("norm {}".format(norm))
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        # print("q norm {}".format(q))
        q[p == 0] = 0
        # print("q norm where p==0 {}".format(q))
        p = smooth_distribution(p, eps)
        # print("p smoothed {}".format(p))
        try:
            q = smooth_distribution(q, eps)
            # print("q smoothed {}".format(q))
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)
        # print("divergence on smoothed p and q {}".format(divergence))
    # print("converse divergence on smoothed p and q {}".format(divergence))
    min_divergence_idx = np.argmin(divergence)
    # print("min_divergence_idx {}".format(min_divergence_idx))
    # print(divergence)
    # print(thresholds)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    print("KL Divergence Threshold {} , divergence {}".format(opt_th, min_divergence))
    return opt_th, min_divergence

def median_quantile_scaling():
    # IQR = 75th quantile — 25th quantile
    # X_scaled = (X — X.median) / IQR
    ''

def min_max_scaling():
    # X_scaled = (X - Xmin)/(Xmax-Xmin)
    ''

# weights = np.asarray([
#                         [
#                             [2.09, -0.98, 1.48, 0.09],
#                             [0.05, -0.14, -1.08, 2.12],
#                             [-0.91, 1.92, 0, -1.03],
#                             [1.87, 0, 1.53, 1.49]
#                         ]
#                     ], dtype=np.float32)
# kl_divergence_scale(weights, num_bins=2048) # float32 -> int8



