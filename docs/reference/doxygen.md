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
documentation. The docs workflow now generates that HTML and mounts it under
<a href="../../api/">`/api/`</a> in the combined MkDocs site. Doxygen also emits
compact XML metadata, which gives us a clean path to a future native MkDocs
import.

## Automated integration path

The first public integration keeps Doxygen HTML authoritative and mounts it
beside MkDocs:

1. `scripts/docs/build_combined_docs.sh --install-doxygen` ensures Doxygen
  1.9.6 is available, then runs `./Documentation/Doxygen/gen_doc.sh -s`.
2. The same helper runs `mkdocs build --strict` to build the HELIA-styled site.
3. The helper calls `scripts/docs/mount_doxygen_html.sh` to copy
  `Documentation/html/` into `site/api/`.
4. The helper verifies `site/api/index.html` exists before the workflow uploads
  the combined `site/` artifact for review or deployment.

The same command works locally when network access is available:

```bash
bash scripts/docs/build_combined_docs.sh --install-doxygen
python3 -m http.server 8012 --directory site
```

This is intentionally conservative: it preserves Arm/CMSIS attribution and the
existing Doxygen output exactly. The visual style differs from MkDocs, but the
API reference remains complete and generated from the repository source.

<p><a class="md-button md-button--primary" href="../../api/">Open generated C API reference</a></p>

## Current limits

The MkDocs site does not yet render individual C functions, structs, or Doxygen
groups natively. Use the mounted Doxygen HTML for exact signatures and
per-kernel behavior while we validate whether XML-driven MkDocs pages can cover
the CMSIS-NN macro-heavy headers reliably.

The current MkDocs pages are intentionally high-level:

- [Operator & Kernel Coverage](../guides/operator-kernel-coverage.md) explains
  the coverage shape by operator family.
- [DSP/MVE Coverage](../guides/dsp-mve-coverage.md) explains why Ambiq focuses
  on MVE, DSP, and graph glue operators.
- This page documents how the generated API reference should be connected to
  the MkDocs site.

## Integration roadmap

| Phase | Goal | Notes |
|---|---|---|
| 1 | Publish generated Doxygen HTML under the same Pages site. | Implemented as `/api/` in the combined docs artifact. |
| 2 | Deploy the combined MkDocs + Doxygen artifact to Pages. | Requires deciding when to enable the gated deploy job. |
| 3 | Validate Doxygen XML in CI. | Confirms the macro-heavy CMSIS-NN headers emit complete, stable metadata. |
| 4 | Prototype native MkDocs API pages from XML. | Candidate approaches include `mkdoxy` or a small purpose-built XML importer. |
| 5 | Generate coverage indexes from XML/source metadata. | Cross-link operator families to exact functions, files, dtypes, and acceleration paths. |

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
