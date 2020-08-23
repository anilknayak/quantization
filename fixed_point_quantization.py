import numpy as np
import loss


# 1. Suppose that you have a layer with outputs in the range of [-a, a).
#    a is a real number could be float32 or float64.
# 2. Multiplying two 8-bit integers is a 16-bit integer
#    Multiplying two 32-bit float is a 64-bit float
# https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks


class FixedPointQuantizationStrategy():
    def __init__(self, from_dtype, to_dtype, verbose=0):
        self.verbose = verbose
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        self.supported_to_dtype = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
        if type(self.to_dtype) == str and self.to_dtype != 'int4' or self.to_dtype not in self.supported_to_dtype:
            raise Exception("Conversion to dtype {} is not supported".format(self.to_dtype))

        self.scale = 1
        self.decide_scale()

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
        max_tensor_value = np.max(abs(self.values))
        max_tensor_nearest_value = np.ceil(max_tensor_value)
        self.scale_factor = max_tensor_nearest_value

    def decide_scale(self):
        if type(self.to_dtype) == str and self.to_dtype != 'int4':
            self.scale = 8
            quantization_bits = 4
            quantization_dtype = 'int4'
            self.to_dtype = np.int8
        if type(self.to_dtype) == str and self.to_dtype != 'uint4':
            self.scale = 15
            quantization_bits = 4
            quantization_dtype = 'uint4'
            self.to_dtype = np.uint8
        else:
            iinfo = np.iinfo(self.to_dtype)
            self.scale = max([abs(iinfo.min), abs(iinfo.max)])
            quantization_bits = iinfo.bits
            quantization_dtype = iinfo.dtype
        finfo = np.finfo(self.from_dtype)
        if self.verbose:
            print("Quantization will be done as follows")
            print("============== From =================")
            print("Dtype: {} Bits: {}".format(finfo.dtype, finfo.bits))
            print("============== To =================")
            print("Dtype: {} Bits: {}".format(quantization_dtype, quantization_bits))
            print("===================================")

    def quantize(self):
        if self.values is None:
            raise Exception("No value for quantization_research")
        if self.values.dtype != self.from_dtype:
            raise Exception("Dtype of from tensorflow value does not match to provided from dtype")
        scaled_values = self.scale * self.values / self.scale_factor
        self.values_quantized = np.floor(scaled_values).astype(self.to_dtype)
        self.quantized = True
        self.dequantize()

    def dequantize(self):
        if not self.quantized:
            raise Exception("Quantization is not performed")
        self.values_dequantized = self.values_quantized * self.scale_factor / self.scale

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
        self.after_dequantization_output = self.after_quantization_output * \
                                           self.quant_weight_strategy.scale_factor / self.descale_factor
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


quant_weight_strategy = FixedPointQuantizationStrategy(np.float64, np.int8)
quant_activation_strategy = FixedPointQuantizationStrategy(np.float64, np.int8)

weights = np.asarray([
    [-0.18120981, -0.29043840],
    [0.49722983, 0.22141714]
], dtype=np.float64)
activations = np.asarray([
    [0.77412377],
    [0.49299395]
], dtype=np.float64)

infer = Inference(quant_weight_strategy,
                  quant_activation_strategy,
                  weights,
                  activations)
infer.forward_before_quantization()
infer.quantize()
infer.forward_after_quantization()
infer.information_loss(1)
