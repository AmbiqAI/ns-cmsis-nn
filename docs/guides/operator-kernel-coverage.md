# Operator & Kernel Coverage

heliaCORE is organized around **operator families** and **kernel variants**.
Operators are the public neural-network operations users care about (Conv2D,
Pad, Softmax, LeakyReLU, and so on). Kernel variants are the implementations
selected for a target CPU, data type, and acceleration path.

The goal is not to document every function on this page. For exact prototypes,
parameters, scratch-buffer queries, and per-function behavior, use the generated
API reference described in [API](../reference/api-groups.md).

## Acceleration paths

Kernel selection is compile-time and follows the feature flags produced by the
compiler and build system for Ambiq targets.

| Path | Typical Ambiq use | Notes |
|---|---|---|
| **Pure C** | Baseline path for smallest Cortex-M targets. | Always available; keeps functionality portable inside Ambiq builds. |
| **DSP** | Apollo-class Cortex-M targets with DSP extension. | Uses SIMD-style DSP intrinsics where they help latency and code size. |
| **MVE / Helium** | Cortex-M55-class Apollo targets. | Primary acceleration focus where MVE is present; vectorized kernels target real HELIA workloads. |

## Operator families

The table below is intentionally grouped. It is meant to help users understand
coverage shape without duplicating the full API reference.

| Family | Examples | Coverage notes |
|---|---|---|
| Convolution | Conv2D, DepthwiseConv2D, TransposeConv2D | int8 is the common path; selected operators also include int16 and int4-weight variants. |
| Dense / matrix | Fully Connected, Batch MatMul, SVDF | Includes helper kernels under `NNSupportFunctions/` for matmul and requantization paths. |
| Recurrent | LSTM, SVDF | Used by speech and sequence workloads in Ambiq model flows. |
| Activation | ReLU, ReLU6, LeakyReLU, PReLU, Hard-Swish, Logistic, Tanh, Clamp | Includes Ambiq coverage for glue operators that appear frequently in field-like models. |
| Elementwise / math | Add, Sub, Mul, Min/Max, Abs, Squared Difference, Sqrt/Rsqrt, Mean | Often small individually, but can matter for end-to-end latency. |
| Comparison / reduction | Equal/NotEqual/Less/Greater, ArgMin/ArgMax, Reduce Min/Max | Maintains quantized operator coverage used by embedded inference graphs. |
| Pooling / softmax | MaxPool, AvgPool, Softmax | Common post-convolution and classifier tail operations. |
| Data movement | Pad, Transpose, Reshape, Concatenation, Split, StridedSlice, Gather/GatherND | Important in real graphs; these operators can dominate if left unoptimized. |
| Quantization | Quantize, Dequantize, requantization helpers | Supports TFLM-style affine quantization and HELIA deployment paths. |
| Layout transforms | Resize Nearest Neighbor, Space-to-Batch, Batch-to-Space, Space-to-Depth, Depth-to-Space | Provides coverage for model-conversion and graph-shaping patterns. |

## Data types

heliaCORE focuses on quantized inference for Ambiq devices:

- **int8 activations / weights** for the broadest operator surface.
- **int16 activations** where models need wider activation precision.
- **int4 weights with int8 activations** for selected high-value kernels such as
  Conv2D, DepthwiseConv2D, and Fully Connected.

Exact dtype support is per function and per variant. Check the generated API
reference before depending on a specific dtype/backend combination.

## Reading the source tree

The source layout mirrors operator families:

| Directory | Purpose |
|---|---|
| `Source/ConvolutionFunctions/` | Conv-family kernels and variants. |
| `Source/FullyConnectedFunctions/` | Dense layer kernels. |
| `Source/ActivationFunctions/` | Activation and clamp-style operators. |
| `Source/NNSupportFunctions/` | Shared matmul, requantization, and helper kernels. |
| `Source/PadFunctions/`, `Source/TransposeFunctions/`, `Source/ReshapeFunctions/` | Data-movement and graph-shaping operators. |

The public C API is declared primarily in `Include/arm_nnfunctions.h` and helper
APIs in `Include/arm_nnsupportfunctions.h`.

## Scope

heliaCORE is intended for HELIA AI workflows on Ambiq silicon. Arm CMSIS-NN
remains the vendor-neutral upstream ecosystem reference for general Cortex-M
kernel development.
