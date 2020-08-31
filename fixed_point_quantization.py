import numpy as np
import loss
import kl_divergence_quantization

# 1. Suppose that you have a layer with outputs in the range of [-a, a).
#    a is a real number could be float32 or float64.
# 2. Multiplying two 8-bit integers is a 16-bit integer
#    Multiplying two 32-bit float is a 64-bit float
# https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks

class FixedPointQuantizationStrategy():
    def __init__(self, from_dtype, to_dtype, verbose=0, scale_type='normal'):
        self.verbose = verbose
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        self.scale_type = scale_type
        self.supported_to_dtype = ['int4', np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
        if self.to_dtype not in self.supported_to_dtype:
            raise Exception("Conversion to dtype {} is not supported".format(self.to_dtype))

        self.quantization_dtype = ''
        self.scale = 1
        self.values = None
        self.scale_factor = 0
        self.zero_center = 0
        self.values_quantized = None
        self.values_dequantized = None
        self.quantized = False

    def set_verbose(self, val):
        self.verbose = val

    def set_values(self, values):
        self.values = values
        # self.convert_zero_mean()
        max_tensor_value = np.max(abs(self.values))
        max_tensor_nearest_value = np.ceil(max_tensor_value)
        self.decide_scale()
        # This scale factor calculated bu analyzing the range of the real values
        # but this has to be done post training.
        # 1. Run the inference and find the best scale factor / range for real value
        # 2. Then try to use that scale factor instead of fixed real valued scale factor.
        if self.scale_type == 'kl':
            threshold, divergence = kl_divergence_quantization.kl_divergence_scale(self.values,
                                                           quantized_dtype='int8',
                                                           num_bins=8001,
                                                           eps=0.0001)
            self.scale_factor = self.scale / threshold
        else:
            self.scale_factor = self.scale / max_tensor_nearest_value

        print("Scale Factor :", self.scale_factor)

    def convert_zero_mean(self):
        self.zero_center = np.mean(self.values)
        self.values -= self.zero_center

    def decide_scale(self):
        if type(self.to_dtype) == str and self.to_dtype == 'int4':
            self.scale = 8
            quantization_bits = 4
            self.quantization_dtype = 'int4'
            self.to_dtype = 'int4'
        elif type(self.to_dtype) == str and self.to_dtype == 'uint4':
            self.scale = 16
            quantization_bits = 4
            self.quantization_dtype = 'uint4'
            self.to_dtype = np.uint8
        else:
            iinfo = np.iinfo(self.to_dtype)
            self.scale = max([abs(iinfo.min), abs(iinfo.max)])-1
            quantization_bits = iinfo.bits
            self.quantization_dtype = iinfo.dtype
        finfo = np.finfo(self.from_dtype)
        if self.verbose:
            print("Quantization will be done as follows")
            print("============== From =================")
            print("Dtype: {} Bits: {}".format(finfo.dtype, finfo.bits))
            print("============== To =================")
            print("Dtype: {} Bits: {} Range: {}".format(self.quantization_dtype,
                                                        quantization_bits, self.scale))
            print("===================================")

    def quantize(self):
        if self.values is None:
            raise Exception("No value for quantization_research")
        if self.values.dtype != self.from_dtype:
            raise Exception("Dtype of from tensorflow value does not match to provided from dtype")
        scaled_values = self.values * self.scale_factor
        to_dtype = self.to_dtype
        if type(self.to_dtype) == str:
            to_dtype = np.int8
        self.values_quantized = np.asarray(np.round(scaled_values), dtype=to_dtype)
        if self.verbose:
            print("Quantized value: ", self.values_quantized)
        self.quantized = True
        self.dequantize()

    def dequantize(self):
        if not self.quantized:
            raise Exception("Quantization is not performed")
        self.values_dequantized = self.values_quantized / self.scale_factor

    def information_loss(self):
        if not self.quantized:
            raise Exception("Quantization is not performed")
        loss.MAE(self.values, self.values_dequantized)
        loss.MSE(self.values, self.values_dequantized)

class Inference():
    def __init__(self, quant_weight_strategy: FixedPointQuantizationStrategy,
                 quant_activation_strategy: FixedPointQuantizationStrategy,
                 weights: np.ndarray,
                 activations: np.ndarray):
        self.quant_weight_strategy = quant_weight_strategy
        self.quant_activation_strategy = quant_activation_strategy
        self.quant_weight_strategy.set_values(weights)
        self.quant_activation_strategy.set_values(activations)
        dtypes_map = {
            'int4': np.int8,
            np.int8: np.int16,
            np.int16: np.int32,
            np.int32: np.int64
        }
        self.inference_quant_output_dtype = dtypes_map[self.quant_weight_strategy.to_dtype]
        iinfo = np.iinfo(self.inference_quant_output_dtype)
        self.descale_factor = max([abs(iinfo.min), abs(iinfo.max)]) // 2
        self.before_quantization_output = None
        self.after_quantization_output = None
        self.after_dequantization_output = None

    def forward_before_quantization(self):
        self.before_quantization_output = np.matmul(self.quant_weight_strategy.values,
                                                    self.quant_activation_strategy.values)
        print("Before quantization_research inference output \n", self.before_quantization_output)

    def quantize(self, verbose=0):
        self.quant_weight_strategy.set_verbose(verbose)
        self.quant_weight_strategy.quantize()
        self.quant_activation_strategy.set_verbose(verbose)
        self.quant_activation_strategy.quantize()

    def forward_after_quantization(self):
        if not (self.quant_weight_strategy.quantized and self.quant_activation_strategy.quantized):
            raise Exception("Quantization is not performed")
        self.after_quantization_output = np.matmul(self.quant_weight_strategy.values_quantized,
                                                   self.quant_activation_strategy.values_quantized,
                                                   dtype=self.inference_quant_output_dtype)
        dequantized_output = self.after_quantization_output / (self.quant_weight_strategy.scale_factor *
                                                               self.quant_activation_strategy.scale_factor)
        self.after_dequantization_output = dequantized_output
        print("After quantization_research inference output \n", self.after_dequantization_output)

    def information_loss(self, verbose=0):
        if self.before_quantization_output is None or \
                self.after_dequantization_output is None:
            raise Exception("Run the inference before and after quantization_research")

        if verbose:
            print("=========== Weight Quantization Information loss ============")
            self.quant_weight_strategy.information_loss()
            print("=========== Activation Quantization Information loss ============")
            self.quant_activation_strategy.information_loss()

        print("=========== Inference information loss ============")
        loss.MAE(self.before_quantization_output, self.after_dequantization_output)
        loss.MSE(self.before_quantization_output, self.after_dequantization_output)

def test(scale_type='kl'):
    from_dtype = np.float32
    # to_dtype = 'int4'
    to_dtype = np.int8

    quant_weight1_strategy = FixedPointQuantizationStrategy(from_dtype, to_dtype, scale_type=scale_type, verbose=1)
    quant_weight2_strategy = FixedPointQuantizationStrategy(from_dtype, to_dtype, scale_type=scale_type, verbose=1)

    weights1 = np.asarray([
        [-1.54, 0.22],
        [-0.26, 0.65]
    ], dtype=from_dtype)
    weights2 = np.asarray([
        [0.35],
        [-0.51]
    ], dtype=from_dtype)

    infer = Inference(quant_weight1_strategy,
                      quant_weight2_strategy,
                      weights1,
                      weights2)
    infer.forward_before_quantization()
    infer.quantize(0)
    infer.forward_after_quantization()
    infer.information_loss(1)


test('kl') # KL divergence to find the best threshold
test('normal') # Max absolute range as threshold