# heliaCORE

> Ambiq's optimized neural-network kernel library for Ambiq silicon (package:
> `ns-cmsis-nn`), built on the Arm CMSIS-NN and CMSIS-Pack ecosystem.

heliaCORE provides quantized neural-network kernels for Ambiq Apollo-class
Cortex-M DSP/MVE targets. It preserves CMSIS-NN-compatible APIs where that
surface applies, adds experimental inherited CMSIS-NN `float32` and `float16`
APIs, and adds Ambiq-tuned operators, kernel variants, and integration paths
for HELIA AI workflows.

[upstream]: https://github.com/ARM-software/CMSIS-NN

**Highlights**

- CMSIS-NN-compatible public APIs where inherited and supported, with additional
  Ambiq/HELIA kernels for Ambiq silicon.
- int8 / int16 / int4-weight quantized kernels for Conv, Depthwise Conv,
  Transpose Conv, Fully Connected, LSTM, SVDF, Pooling, Softmax, elementwise
  math and more.
- Experimental `float32` and `float16` CMSIS-NN APIs are available for selected
  operators when explicitly enabled.
- Three backend paths selected automatically at build time from your toolchain
  CPU flags: **pure C**, **DSP**, and **MVE / Helium**. MVE is a primary
  optimization target where supported by the Ambiq device.
- Distributed as a **CMSIS-Pack**, a **Zephyr module**, CMake package/tarball,
  and HELIA/neuralSPOT-X integration point.
- No dynamic allocation; the caller owns every buffer.
- Apache-2.0 on upstream-derived files; Ambiq Apollo SDK License on Ambiq
  additions (see [License](#license)).

---

## What heliaCORE is

- Ambiq's foundation neural-network kernel layer for HELIA AI workflows on
  Ambiq silicon.
- Built on Arm CMSIS-NN concepts and compatible API surfaces where applicable,
  with attribution and license preservation for inherited sources.
- Optimized for Ambiq Apollo SoCs, taking advantage of M-Profile Vector
  Extensions (MVE / Helium) and DSP instructions where available.
- A kernel backend used by runtime and compiler flows on Ambiq parts. It follows
  the [int8][quant-int8] and int16 quantization specifications used by TFLM.

[tflm]: https://www.tensorflow.org/lite/microcontrollers
[quant-int8]: https://www.tensorflow.org/lite/performance/quantization_spec
[executorch]: https://executorch.ai/

## What heliaCORE is *not*

- **Not** a vendor-neutral CMSIS-NN distribution. heliaCORE is intended for
  HELIA AI workflows on Ambiq silicon. For general Cortex-M kernel work, use
  [upstream Arm CMSIS-NN][upstream].
- **Not** a model runtime. Pair it with TFLM (or your own runtime) to execute
  models.

## License

heliaCORE ships under two license files:

| File | Applies to | SPDX |
|---|---|---|
| [`LICENSE`](LICENSE) | The fork as a whole — restricted to use on Ambiq CPUs | `LicenseRef-Ambiq-Apollo-SDK` |
| [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) | Files originating from upstream Arm CMSIS-NN | `Apache-2.0` |

Per-file copyright and license is declared via SPDX headers. The upstream
CMSIS-NN copyright and Apache-2.0 license terms are preserved on every file
that originated upstream; modifications are dual-attributed to Ambiq.

See [`NOTICE`](NOTICE) for the full attribution and licensing summary.

## Relationship to upstream Arm CMSIS-NN

heliaCORE was forked from [`ARM-software/CMSIS-NN`][upstream] and is maintained
as an Ambiq silicon-focused kernel library. We:

- Preserve inherited CMSIS-NN-compatible public APIs where supported so existing
  integrations can continue to use familiar `arm_*` entry points.
- Preserve upstream copyright, SPDX, and Apache-2.0 license notices on inherited
  files.
- Periodically review upstream for relevant kernels and improvements.
- Add Ambiq-specific kernels, variants, and Apollo optimizations for HELIA AI
  workflows.

If you find a kernel here that does not exist in upstream, that is intentional
and part of Ambiq's HELIA-focused coverage for Ambiq devices.

### Detecting heliaCORE at compile time

Because heliaCORE ships as `libcmsis-nn.a` and exposes a `cmsis-nn` CMake
ALIAS for TFLM build compatibility, downstream code that calls Ambiq-only
kernels (e.g. `arm_gather_s8`) should guard against accidentally being built
against another CMSIS-NN implementation.
[`Include/arm_nn_types.h`](Include/arm_nn_types.h) defines:

| Macro | Meaning |
|---|---|
| `NS_CMSIS_NN` | Always `1` when this header is from heliaCORE NN. |
| `NS_CMSIS_NN_VERSION_MAJOR` / `MINOR` / `PATCH` | Component version numbers. |
| `NS_CMSIS_NN_VERSION` | Packed as `MAJOR * 1000000 + MINOR * 1000 + PATCH` (3-digit fields, so semantic ordering is preserved for any reasonable version). |

Recommended guard:

```c
#include "arm_nn_types.h"

#if !defined(NS_CMSIS_NN)
#  error "this code requires ns-cmsis-nn"
#endif
#if NS_CMSIS_NN_VERSION < 7024000
#  error "needs ns-cmsis-nn >= 7.24.0"
#endif
```

## Naming

| Name | Used for |
|---|---|
| **heliaCORE** | Product / brand name (prose, docs, marketing) |
| `ns-cmsis-nn` | Package id, repo name, CMake/Zephyr module name, file/path identifiers |
| `arm_*` C symbols | Public API — kept identical to upstream |

## Repository layout

```
ns-cmsis-nn/
├── Include/                       Public headers (C API)
│   ├── arm_nnfunctions.h          Top-level kernel API
│   ├── arm_nnsupportfunctions.h   Helpers shared between kernels
│   ├── arm_nn_types.h             Shared types (dims, params, context)
│   ├── arm_nn_math_types.h        Quantization / fixed-point types
│   ├── arm_nn_tables.h            Lookup tables (e.g. for activations)
│   └── Internal/                  Compiler / intrinsics compatibility shim
├── Source/                        Kernel implementations (one .c per function)
│   ├── ActivationFunctions/
│   ├── BasicMathFunctions/
│   ├── ComparisonFunctions/
│   ├── ConcatenationFunctions/
│   ├── ConvolutionFunctions/
│   ├── FullyConnectedFunctions/
│   ├── GatherFunctions/
│   ├── LSTMFunctions/
│   ├── NNSupportFunctions/        Matmul kernels, requantize, ...
│   ├── PadFunctions/
│   ├── PoolingFunctions/
│   ├── QuantizationFunctions/
│   ├── ReshapeFunctions/
│   ├── SoftmaxFunctions/
│   ├── StridedSliceFunctions/
│   ├── SVDFunctions/
│   └── TransposeFunctions/
├── Tests/UnitTest/                Per-kernel unit tests (Unity + Python harness)
├── Documentation/                 Doxygen sources + pre-built HTML
├── Examples/                      Sample integrations
├── zephyr/                        Zephyr module manifest + Kconfig + CMake glue
├── Ambiq.NS-CMSIS-NN.pdsc         CMSIS-Pack description
├── CMakeLists.txt                 Top-level CMake (defines `cmsis-nn` static lib)
├── LICENSE                        Ambiq Apollo SDK License
├── LICENSES/                      SPDX-referenced license texts (Apache-2.0, Ambiq)
└── NOTICE                         Attribution + dual-license summary
```

---

## Operator support

Optimizations are picked at compile time based on the architecture features the
compiler reports.

| Backend | When it is used |
|---|---|
| **Pure C** | Always available; used on Cortex-M0 / M3 (no DSP, no MVE) |
| **DSP** | Cortex-M4, M7, M33 (with DSP extension) — uses SIMD intrinsics |
| **MVE** | Cortex-M55, M85 — uses Arm Helium / M-Profile Vector Extension |

### Experimental floating-point support

heliaCORE NN includes the experimental CMSIS-NN `float32` and `float16` APIs
inherited from upstream. They are disabled by default so integer-only builds do
not pay for extra code size or public API surface. Enable them only for
applications that strictly need floating-point kernels.

The floating-point API follows the same CMSIS-NN style as the integer API,
including the TFLM-shaped parameter structures. This keeps the public surface
consistent across data types. `float16` APIs are included even though TFLM does
not define a `float16` operator contract; they are intended for frameworks that
can carry `float16` operator flows, such as [ExecuTorch][executorch].

Floating-point support primarily targets Cortex-M CPUs with Helium/MVE and
hardware floating-point support. Pure C scalar paths are present for
correctness, bring-up, and fallback, but they are not the main performance
target. On cores that only provide the classic DSP extension, float kernels may
compile through the scalar C path but are not a performance target.

For float operators that support `arm_nn_weight_format_flt`, MVE performance is
generally better when constant weights are provided in the packed `NTxN` layout
instead of the standard `NT x T` layout. This avoids the gather-heavy RHS access
pattern of the standard formulation and is the preferred deployment format when
offline repacking is available.

The scalar floating-point code can also be compiled for Arm A-class CPUs with
`float16` support and may benefit from NEON or SVE auto-vectorization. That is
not an intended deployment target for heliaCORE NN float support, and the
resulting performance is expected to be suboptimal compared with libraries
designed for that class of processor. For Arm A-class CPUs, prefer optimized
inference libraries such as Arm Compute Library or XNNPACK.

### Operator coverage matrix

`Yes` = available; `No` = not implemented; `N/A` = does not apply to that dtype.
`*` int4 = int4 weights with int8 activations.

| Operator                       | C int8 | C int16 | C int4* | DSP int8 | DSP int16 | DSP int4* | MVE int8 | MVE int16 | MVE int4* |
|--------------------------------|:------:|:-------:|:-------:|:--------:|:---------:|:---------:|:--------:|:---------:|:---------:|
| Conv2D                         | Yes    | Yes     | Yes     | Yes      | Yes       | Yes       | Yes      | Yes       | Yes       |
| DepthwiseConv2D                | Yes    | Yes     | Yes     | Yes      | Yes       | Yes       | Yes      | Yes       | Yes       |
| TransposeConv2D                | Yes    | No      | No      | Yes      | No        | No        | Yes      | No        | No        |
| Fully Connected                | Yes    | Yes     | Yes     | Yes      | Yes       | Yes       | Yes      | Yes       | Yes       |
| Batch MatMul                   | Yes    | Yes     | No      | Yes      | Yes       | No        | Yes      | Yes       | No        |
| Add / Sub                      | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Mul                            | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Minimum / Maximum              | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Abs                            | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Squared Difference             | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Sqrt / Rsqrt                   | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Mean                           | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Reduce Min / Reduce Max        | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| ArgMin / ArgMax                | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Comparison (==, !=, <, >, ...) | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| ReLU / ReLU6                   | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Leaky ReLU                     | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| PReLU                          | Yes    | No      | N/A     | Yes      | No        | N/A       | Yes      | No        | N/A       |
| Hard-Swish                     | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Logistic                       | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Tanh                           | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Clamp                          | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Softmax                        | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| MaxPool / AvgPool              | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Pad                            | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Transpose                      | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Reshape                        | Yes    | N/A     | N/A     | Yes      | N/A       | N/A       | Yes      | N/A       | N/A       |
| Concatenation                  | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Split                          | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| StridedSlice                   | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Gather / GatherND              | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Quantize / Dequantize          | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Resize Nearest Neighbor        | Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Space-to-Batch / Batch-to-Space| Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| Space-to-Depth / Depth-to-Space| Yes    | Yes     | N/A     | Yes      | Yes       | N/A       | Yes      | Yes       | N/A       |
| LSTM (unidirectional)          | Yes    | Yes     | No      | Yes      | Yes       | No        | Yes      | Yes       | No        |
| SVDF                           | Yes    | No      | No      | Yes      | No        | No        | Yes      | No        | No        |

> Coverage above reflects what is shipped in `Source/` today. For exact dtype
> support per kernel, see the function prototypes in
> [`Include/arm_nnfunctions.h`](Include/arm_nnfunctions.h).

### Experimental float operator coverage

The table below summarizes public `float16` / `float32` operator coverage. The
MVE column means the implementation contains explicit Helium-specialized paths
for at least the supported common cases; scalar C fallback remains available
unless the target or toolchain cannot provide the required floating-point type.

| Operator | C float16 | C float32 | MVE float16 | MVE float32 |
|---|:---:|:---:|:---:|:---:|
| Conv2D | Yes | Yes | Yes | Yes |
| DepthwiseConv2D | Yes | Yes | Yes | Yes |
| TransposeConv2D | Yes | Yes | Yes | Yes |
| Fully Connected | Yes | Yes | Yes | Yes |
| Batch MatMul | Yes | Yes | Yes | Yes |
| Activation | Yes | Yes | Yes | Yes |
| Add | Yes | Yes | Yes | Yes |
| Minimum / Maximum | Yes | Yes | Yes | Yes |
| Mul | Yes | Yes | Yes | Yes |
| MaxPool / AvgPool | Yes | Yes | Yes | Yes |
| Softmax | Yes | Yes | Yes | Yes |
| LSTM (unidirectional) | Yes | Yes | Yes | Yes |
| SVDF | Yes | Yes | Yes | Yes |
| Batch Norm | Yes | Yes | Yes | Yes |
| Pad | Yes | Yes | No | No |
| Transpose | Yes | Yes | Yes | Yes |
| Reshape | Yes | Yes | No | No |
| Concatenation | Yes | Yes | No | No |
| Dequantize | N/A | Yes | N/A | Yes |

---

## Architecture & data layout

- **Data layout.** Activations are laid out **NHWC** (batch, height, width,
  channel). Weight layouts are kernel-specific and documented per function;
  most convolutions use HWIO with per-channel scaling.
- **Quantization.** TFLM-style affine quantization. int8 or int16 activations,
  int8 (or int4-packed) weights, int32 bias. Conv-family kernels accept
  per-channel weight scales; fully-connected accepts per-tensor or per-channel
  depending on the variant. The exact scheme each kernel implements matches
  the [TFLM int8 quantization spec][quant-int8].
- **Buffer convention.** Every kernel takes a `cmsis_nn_context` whose `buf`
  must be sized via the matching `arm_*_get_buffer_size*` query. If the query
  returns 0, you may pass `{ NULL, 0 }`.
- **Backend selection.** Compile-time, driven by preprocessor defines:
  - `ARM_MATH_MVEI` → Helium / MVE path
  - `ARM_MATH_DSP` (without MVE) → DSP intrinsics path
  - neither → pure C

  The CMake build sets these automatically from your toolchain CPU flags.
- **No dynamic allocation.** Kernels never call `malloc` / `free`. The caller
  owns all input, output, weight, bias and scratch buffers.
- **Threading.** Kernels are reentrant on disjoint buffers; there is no
  internal global state to protect.

---

## Getting started

heliaCORE NN can be consumed three ways. Pick whichever fits your build system.

### Using with TensorFlow Lite for Microcontrollers

The most common consumer of these kernels is TFLM. Because heliaCORE NN keeps
the upstream `arm_*` C ABI, TFLM's existing CMSIS-NN integration links against
it unchanged — point your build at this repository instead of
`ARM-software/CMSIS-NN` and rebuild. On Ambiq parts this is wired up for you
by the Ambiq SDK; you generally do not need to integrate it by hand.

### As a CMSIS-Pack

The pack is published as a release asset on this repository:
[`Ambiq.NS-CMSIS-NN.<version>.pack`](https://github.com/AmbiqAI/ns-cmsis-nn/releases/latest).
Install via `cpackget`:

```sh
cpackget add Ambiq.NS-CMSIS-NN.<version>.pack
```

Then add the `CMSIS:NN Lib` component in your `*.csolution.yml` /
`*.cproject.yml`.

### As a Zephyr module

Add to your `west.yml`:

```yaml
manifest:
  projects:
    - name: ns-cmsis-nn
      url: https://github.com/AmbiqAI/ns-cmsis-nn
      revision: main
      path: modules/lib/ns-cmsis-nn
```

Then enable in your project's `prj.conf`:

```kconfig
CONFIG_NS_CMSIS_NN=y
CONFIG_NS_CMSIS_NN_ALL=y     # or pick individual kernel groups
```

Per-group `CONFIG_NS_CMSIS_NN_*` symbols are defined in
[`zephyr/Kconfig`](zephyr/Kconfig). The Zephyr module is gated by
`CONFIG_CPU_CORTEX_M` and is mutually exclusive with upstream `CMSIS_NN`.

### As a CMake subdirectory

```cmake
add_subdirectory(third_party/ns-cmsis-nn)
target_link_libraries(my_app PRIVATE cmsis-nn)
```

The CMake build picks up the right backend (C / DSP / MVE) from your
toolchain's CPU flags. See
[`CMakeLists.txt`](CMakeLists.txt) for the build target definition, and
[`Documentation/build.md`](Documentation/build.md) for a full reference
of the SSoT CMake module — including how to opt in/out of operator
groups and how to add a new one.

### Calling a kernel directly

Minimal example — a quantized 64→10 fully-connected layer:

```c
#include "arm_nnfunctions.h"

static const int8_t  weights[10 * 64];   /* trained, int8 */
static const int32_t bias[10];           /* int32 */
static int8_t        output[10];

arm_cmsis_nn_status run_fc(const int8_t *input)
{
    cmsis_nn_dims input_dims  = { .n = 1, .h = 1, .w = 1, .c = 64 };
    cmsis_nn_dims filter_dims = { .n = 10, .h = 1, .w = 1, .c = 64 };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = 10 };
    cmsis_nn_dims output_dims = { .n = 1, .h = 1, .w = 1, .c = 10 };

    cmsis_nn_fc_params fc_params = {
        .input_offset  =  128,           /* zero-points from your quantizer */
        .filter_offset =    0,
        .output_offset =   -2,
        .activation    = { .min = INT8_MIN, .max = INT8_MAX },
    };
    cmsis_nn_per_tensor_quant_params q = {
        .multiplier = 1073741824,        /* M0 from your quantizer */
        .shift      = -7,                /* shift from your quantizer */
    };

    int32_t buf_sz = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    int8_t  scratch[buf_sz];             /* or pool / static buffer */
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_sz };

    return arm_fully_connected_s8(&ctx, &fc_params, &q,
                                  &input_dims,  input,
                                  &filter_dims, weights,
                                  &bias_dims,   bias,
                                  &output_dims, output);
}
```

In practice, you don't write this by hand — TFLM (or another runtime) emits
the parameter structs from the quantized model. The example shows the call
shape every kernel in the library follows.

---

## Build & test

The repo ships a devcontainer with all required tooling
(see [`.devcontainer/`](.devcontainer/)). To build the library standalone:

```sh
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<your-toolchain.cmake> -DTARGET_CPU=cortex-m55
make
```

Default optimization is `-Ofast`. Override with `-DCMSIS_OPTIMIZATION_LEVEL=-O2`.
Note that with `-O0`, you must define `ARM_MATH_AUTOVECTORIZE` for Helium parts.

Compile-time options that affect headers (set the same flag in TFLM):

| Option | Effect |
|---|---|
| `ARM_NN_ENABLE_F32` | Enables experimental `float32` operator support. Leave disabled unless the application needs float32 kernels. |
| `ARM_NN_ENABLE_F16` | Enables experimental `float16` operator support. Leave disabled unless the application needs float16 kernels and the toolchain/target support them. |
| `CMSIS_NN_USE_SINGLE_ROUNDING` | Use single instead of double rounding in requantization. May change outputs. |
| `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` | Inline assembly for `arm_nn_requantize`. Faster on Cortex-M4, slower elsewhere. |
| `OPTIONAL_RESTRICT_KEYWORD=__restrict` | Enables `restrict` on int4/int8 conv outputs. Recommended on Cortex-M7. |

Additional implementation-selection options:

| Option | Effect |
|---|---|
| `NN_DISABLE_SPECIALIZATION` | Disables optional shape/layout-specific fast paths and forces the corresponding generic implementations. Useful for debugging or validating specialized kernels against generic paths. |
| `ARM_NN_USE_EXP_LUT` | Selects the LUT-based scalar float softmax exp approximation. This is the default if no scalar float softmax exp macro is defined. |
| `ARM_NN_USE_EXP_TAYLOR` | Selects the Taylor/Estrin scalar float softmax exp approximation to avoid the extra lookup-table storage. Do not define this with `ARM_NN_USE_EXP_LUT`. |

### Running unit tests

Unit tests live in [`Tests/UnitTest/`](Tests/UnitTest/). See the
[Tests README](Tests/UnitTest/README.md) for the full workflow.

### Supported toolchains

- Arm Compiler 6
- Arm GNU Toolchain (`arm-none-eabi-gcc`)
- LLVM Embedded Toolchain for Arm (ATfE) — best-effort

IAR is currently untested. Compiling for host is not supported out of the box.

---

## Versioning & releases

- Semantic versioning (`MAJOR.MINOR.PATCH`).
- Releases are cut from `main` by [release-please][release-please] driven by
  [Conventional Commits][conv-commits].
- Each release publishes:
  - A GitHub Release with a CMSIS-Pack asset:
    `Ambiq.NS-CMSIS-NN.<version>.pack`.
  - A new entry in [`CHANGELOG.md`](CHANGELOG.md).
  - (Optionally) refreshed Doxygen docs.

[release-please]: https://github.com/googleapis/release-please
[conv-commits]: https://www.conventionalcommits.org

---

## Documentation

API reference is generated with Doxygen and published as a GitHub Pages site
on each release. To build it locally:

```sh
./Documentation/Doxygen/gen_doc.sh
```

Output lands in `Documentation/html/`.

## Roadmap

Public-launch readiness is tracked under the
[GitHub milestones](https://github.com/AmbiqAI/ns-cmsis-nn/milestones).
Issues blocking the public release carry the
[`public-launch`](https://github.com/AmbiqAI/ns-cmsis-nn/labels/public-launch)
label.

## FAQ

**Why not just use upstream Arm CMSIS-NN?**
Upstream Arm CMSIS-NN is the right choice for vendor-neutral Cortex-M work. Use
heliaCORE when you are building HELIA AI workflows for Ambiq silicon and need
Apollo-tuned kernels or Ambiq-specific operator coverage.

**Are the `arm_*` symbols ABI-stable across heliaCORE NN versions?**
Inherited symbols and signatures are kept compatible where supported so TFLM
and other consumers can continue to link through familiar `arm_*` entry points.

**How do I enable the experimental float APIs?**
Enable `ARM_NN_ENABLE_F32` and/or `ARM_NN_ENABLE_F16` in the build. They are
off by default to keep integer-only builds small. If the enabled float headers
are consumed by TFLM or another downstream build, use matching definitions in
that build too.

**Do floating-point kernels target all IEEE edge cases?**
No. For performance reasons, the current floating-point kernels do not
specifically target IEEE edge cases such as `NaN`, `Inf`,
denormals/subnormals, or signed zero. The intended use is that pre-processing
provides finite, numerically safe input data and that model weights and biases
do not contain aberrant values.

**What is the scalar float softmax storage cost?**
The scalar floating-point softmax path uses the LUT-based exp approximation by
default for performance. This adds one 257-entry lookup table per enabled float
precision: about 514 bytes for `float16` and about 1028 bytes for `float32`.
Define `ARM_NN_USE_EXP_TAYLOR` to avoid the lookup-table storage. Do not define
`ARM_NN_USE_EXP_LUT` and `ARM_NN_USE_EXP_TAYLOR` at the same time.

**Can I build the library for the host (x86 / Mac) for testing?**
Not directly — the kernels target Cortex-M. The `Tests/UnitTest/` harness
builds for Cortex-M and runs the binaries under emulation.

**How do I report a bug or request a kernel?**
Open an issue at
<https://github.com/AmbiqAI/ns-cmsis-nn/issues>. Please include the target
core (M4 / M55 / …), the toolchain, and a minimal reproducer if you can.

## Contributing

External contributions are welcome on additive kernels and bug fixes that do
not break inherited CMSIS-NN-compatible APIs or Ambiq HELIA integration paths. A
`CONTRIBUTING.md` with the full workflow is in progress (tracked separately).

In the meantime, the short version:

- One function per file; file name matches the function name.
- Variable and function names: lowercase with underscores.
- Add a Doxygen header to every public prototype.
- Every new kernel ships with a unit test under `Tests/UnitTest/`.

Issues and PRs go to
[`AmbiqAI/ns-cmsis-nn`](https://github.com/AmbiqAI/ns-cmsis-nn/issues).

## Support

For bug reports and questions, please open an issue:
<https://github.com/AmbiqAI/ns-cmsis-nn/issues>.

For commercial / Apollo-platform support, contact Ambiq through your usual
support channel.

## Acknowledgements

heliaCORE NN is built on top of [Arm CMSIS-NN][upstream] (Apache-2.0).
We are grateful to the Arm CMSIS-NN team for the foundational kernel work
and to the TFLM community for the quantization specifications this library
implements.
