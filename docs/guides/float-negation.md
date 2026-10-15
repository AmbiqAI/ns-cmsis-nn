# Floating-point negation

`arm_nn_neg_f32` and `arm_nn_neg_f16` implement elementwise NEG by toggling the IEEE sign bit. They preserve every other bit, including NaN payloads and subnormals, and exchange positive and negative zero. Floating-point flush and default-NaN settings do not affect the result.

The input and output have the same shape and dtype; the caller passes their element count. Exact in-place use is supported. Buffers need natural element alignment and must otherwise not overlap. Zero elements accepts null buffers and writes nothing; negative counts, byte counts above INT32_MAX and null nonempty buffers return `ARM_CMSIS_NN_ARG_ERROR` before writes.

Both entries use portable bit operations on generic and DSP targets. M55 builds use predicated MVE loads and stores, including tails, unless `ARM_MATH_AUTOVECTORIZE` selects compiler vectorization. Each dtype obeys its own float build gate and requires no scratch or other public kernel.
