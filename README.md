# Quantization

I have been reading and doing NN model optimization on weight quantization by clusering and weight pruing.
I have implemented some basic weight quantization. I will be adding more to this repo.

I have been using libraries like TensorFlow for quantization but I would like to know under the hood of quantization.

I might have made some mistakes. If so, please comment and share so that I can fix it and nonethless we will both get to know something amazing about
ongoing research in quantization.

# Fixed point weight quantization
1. float16, float32, float64 -> int16, int8, int4

# Clustering based weight quantization
1. float16, float32, float64 -> int8 [weight indexes] and float32/float16 [cluster centers]

# Other weight quantization
1. Min max weight quantization
2. Threshold/Probability based weight quantization

# Activation quantization
Coming soon

# Hoffman encoding based quantization
Coming soon

# Weight pruning
Coming soon

# LSTM weight quantization


