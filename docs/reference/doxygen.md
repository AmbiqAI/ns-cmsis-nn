# Doxygen API

heliaCORE already carries an inherited Doxygen pipeline under
`Documentation/Doxygen/`. That pipeline generates the detailed C API reference
from the headers and source tree, including inherited CMSIS-NN-compatible APIs
and Ambiq additions.

## Current source of truth

| Item | Location |
|---|---|
| Public kernel API | `Include/arm_nnfunctions.h` |
| Support/helper API | `Include/arm_nnsupportfunctions.h` |
| Shared types | `Include/arm_nn_types.h` |
| Doxygen config | `Documentation/Doxygen/nn.dxy.in` |
| Doxygen entry page | `Documentation/Doxygen/src/mainpage.md` |
| Generation script | `Documentation/Doxygen/gen_doc.sh` |
| Generated HTML output | `Documentation/html/` |
| Generated XML metadata | `Documentation/xml/` |

The existing Doxygen flow remains the source of truth for detailed C API
documentation. It now emits both HTML and compact XML metadata: HTML is the
authoritative human-readable API reference, while XML gives us a clean path to a
future native MkDocs import.

## Recommended integration path

For this MkDocs site, there are two reasonable integration options:

1. **Mount generated Doxygen HTML under `/api/`**
   - Lowest risk.
   - Preserves Arm/CMSIS attribution and the existing Doxygen output exactly.
   - The visual style will differ from MkDocs, but the API reference remains
     complete.

2. **Generate Doxygen XML and render with mkdoxy**
   - Better long-term look and search integration inside MkDocs.
   - Uses the generated XML metadata now emitted by the Doxygen configuration.
   - Still needs validation on the CMSIS-NN macro-heavy headers before we rely
     on it for the public site.

For the first public heliaCORE site, the conservative recommendation is to keep
Doxygen HTML as the authoritative API reference and link to it from MkDocs.
The XML output can be validated in a follow-up PR before deciding whether to
render native MkDocs API pages.

## API groups

The inherited Doxygen main page already groups kernels by functional category:

| Group | Examples |
|---|---|
| Activation | ReLU, LeakyReLU, PReLU, Hard-Swish, Logistic, Tanh, Clamp |
| Elementwise / reduction / comparison | Add/Sub, Mul, Min/Max, Mean, ArgMin/ArgMax, comparison operators |
| Convolution / dense | Conv2D, DepthwiseConv2D, TransposeConv2D, Fully Connected, Batch MatMul |
| Sequence | LSTM, SVDF |
| Data movement | Pad, Concatenation, Reshape, StridedSlice, Transpose, Gather |
| Quantization / classifier tail | Quantize, Dequantize, Pooling, Softmax |

That grouping should remain the backbone of the API reference so users can move
between the high-level coverage guide and exact C function documentation.

## Attribution

Inherited CMSIS-NN comments and documentation remain under their original Arm
license terms. Keep generated API documentation attribution intact when mounting
or transforming Doxygen output.
