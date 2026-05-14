# heliaCORE NN

> Ambiq's NN kernel library (package: `ns-cmsis-nn`) — an Ambiq-optimized fork of Arm CMSIS-NN, tuned for the Apollo family of ultra-low-power SoCs.

heliaCORE NN provides quantized neural-network kernels for Arm Cortex-M class
processors. It is API-compatible with [Arm CMSIS-NN][upstream] and adds
additional operators and Apollo-specific optimizations on top.

[upstream]: https://github.com/ARM-software/CMSIS-NN

---

## What heliaCORE NN is

- A **drop-in superset** of Arm CMSIS-NN. Every `arm_*` symbol in upstream is
  preserved; we add new operators and refine existing kernels.
- Optimized for Ambiq Apollo SoCs, taking advantage of M-Profile Vector
  Extensions (MVE / Helium) and DSP instructions where available.
- The NN backend used by [TensorFlow Lite for Microcontrollers (TFLM)][tflm] on
  Ambiq parts. It follows the [int8][quant-int8] and int16 quantization
  specifications used by TFLM.

[tflm]: https://www.tensorflow.org/lite/microcontrollers
[quant-int8]: https://www.tensorflow.org/lite/performance/quantization_spec

## What heliaCORE NN is *not*

- **Not** a general-purpose CMSIS-NN replacement. The license restricts use to
  software running on Ambiq-manufactured CPUs (see [License](#license) below).
  If you need an unrestricted CMSIS-NN, use [upstream Arm CMSIS-NN][upstream].
- **Not** a model runtime. Pair it with TFLM (or your own runtime) to execute
  models.

## License

heliaCORE NN ships under two license files:

| File | Applies to | SPDX |
|---|---|---|
| [`LICENSE`](LICENSE) | The fork as a whole — restricted to use on Ambiq CPUs | `LicenseRef-Ambiq-Apollo-SDK` |
| [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) | Files originating from upstream Arm CMSIS-NN | `Apache-2.0` |

Per-file copyright and license is declared via SPDX headers. The upstream
CMSIS-NN copyright and Apache-2.0 license terms are preserved on every file
that originated upstream; modifications are dual-attributed to Ambiq.

See [`NOTICE`](NOTICE) for the full attribution and licensing summary.

## Relationship to upstream Arm CMSIS-NN

heliaCORE NN was forked from [`ARM-software/CMSIS-NN`][upstream] and is
maintained as a long-lived fork. We:

- Keep the public C API (`arm_*` functions in `Include/`) **identical** to
  upstream so existing TFLM integrations continue to work unchanged.
- Periodically review upstream for new kernels and pull them in.
- Add new kernels and Apollo-specific optimizations on top.

If you find a kernel here that does not exist in upstream, that is intentional
and part of the heliaCORE NN superset.

## Naming

| Name | Used for |
|---|---|
| **heliaCORE NN** | Product / brand name (prose, docs, marketing) |
| **heliaCORE** | Short brand name when used standalone |
| `ns-cmsis-nn` | Package id, repo name, CMake/Zephyr module name, file/path identifiers |
| `arm_*` C symbols | Public API — kept identical to upstream |

---

## Operator support

Optimizations are picked at compile time based on the architecture features the
compiler reports.

| Backend | When it is used |
|---|---|
| **Pure C** | Always available; used on Cortex-M0 / M3 (no DSP, no MVE) |
| **DSP** | Cortex-M4, M7, M33 (with DSP extension) — uses SIMD intrinsics |
| **MVE** | Cortex-M55, M85 — uses Arm Helium / M-Profile Vector Extension |

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

---

## Getting started

heliaCORE NN can be consumed three ways. Pick whichever fits your build system.

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
[`CMakeLists.txt`](CMakeLists.txt) for the build target definition.

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
| `CMSIS_NN_USE_SINGLE_ROUNDING` | Use single instead of double rounding in requantization. May change outputs. |
| `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` | Inline assembly for `arm_nn_requantize`. Faster on Cortex-M4, slower elsewhere. |
| `OPTIONAL_RESTRICT_KEYWORD=__restrict` | Enables `restrict` on int4/int8 conv outputs. Recommended on Cortex-M7. |

### Running unit tests

Unit tests live in [`Tests/UnitTest/`](Tests/UnitTest/). See the
[Tests README](Tests/UnitTest/README.md) for the full workflow.

### Supported toolchains

- Arm Compiler 6
- Arm GNU Toolchain (`arm-none-eabi-gcc`)
- LLVM Embedded Toolchain for Arm (ATfE) — best-effort

IAR is currently untested. Compiling for host is not supported out of the box.

---

## Documentation

API reference is generated with Doxygen and published as a GitHub Pages site
on each release. To build it locally:

```sh
./Documentation/Doxygen/gen_doc.sh
```

Output lands in `Documentation/html/`.

## Contributing

External contributions are welcome on additive kernels and bug fixes that do
not break the CMSIS-NN superset guarantee. A `CONTRIBUTING.md` with the full
workflow is in progress (tracked separately).

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
