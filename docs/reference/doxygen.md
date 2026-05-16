# API Generation Notes

heliaCORE already carries an inherited Doxygen pipeline under
`Documentation/Doxygen/`. That pipeline generates the detailed C API reference
from the public headers, including inherited CMSIS-NN-compatible APIs and Ambiq
additions.

## Current source of truth

| Item | Location |
|---|---|
| Public kernel API | `Include/arm_nnfunctions.h` |
| Support/helper API | `Include/arm_nnsupportfunctions.h` |
| Shared types | `Include/arm_nn_types.h` |
| Doxygen config | `Documentation/Doxygen/nn.dxy.in` |
| Doxygen entry page | `Documentation/Doxygen/src/mainpage.md` |
| Generation script | `Documentation/Doxygen/gen_doc.sh` |
| Standalone Doxygen HTML byproduct | `Documentation/html/` |
| Generated XML metadata for Sphinx | `Documentation/xml/` |

The existing Doxygen flow remains the source of truth for detailed public C API
documentation. The docs workflow generates XML metadata from the public headers
and renders it into the Sphinx site through Breathe and Exhale, producing a
native <a href="../api/library_root.html">C API reference</a> in the same
navigation, theme, and search experience as the rest of the site. The standalone
Doxygen HTML output is not the customer-facing website.

## Automated integration path

The public docs build keeps Doxygen as the parser and source of truth, then uses
Sphinx for the customer-facing site:

1. `scripts/docs/build_sphinx_docs.sh --install-doxygen` ensures Doxygen 1.9.6
  is available, then runs `./Documentation/Doxygen/gen_doc.sh -s`.
2. Doxygen emits `Documentation/xml/index.xml` and related XML files.
3. `sphinx-build` reads `docs/conf.py`, where Breathe points at the XML output
  and Exhale generates the API page tree under `docs/api/`.
4. The helper verifies `site/api/library_root.html` exists before the workflow
  uploads the `site/` artifact for review or deployment.

For local preview, the one-line helper installs/reuses the Sphinx environment,
builds the site, and starts a static server:

```bash
bash scripts/docs/serve_sphinx_docs.sh
```

Then open `http://127.0.0.1:8014/`. For a faster rebuild after Doxygen XML has
already been generated, use:

```bash
bash scripts/docs/serve_sphinx_docs.sh --skip-doxygen
```

The lower-level build command is the same helper used by CI:

```bash
bash scripts/docs/build_sphinx_docs.sh --install-doxygen
```

This keeps Arm/CMSIS attribution attached to the generated public API comments
while making the API reference part of the same Sphinx site as the guides and
getting started material.

<p><a class="hc-button hc-button-primary" href="../api/library_root.html">Open generated C API reference</a></p>
<p><a class="hc-button" href="api-groups.html">Browse grouped C API</a></p>

## Current limits

The Sphinx site now renders individual C functions, structs, files, and Doxygen
groups from XML. The generated output still depends on how clearly Doxygen can
understand the CMSIS-NN macro-heavy headers, so customer-facing guide pages
remain important for explaining operator coverage and integration paths.

The hand-written Sphinx guide pages are intentionally high-level:

- [Operator & Kernel Coverage](../guides/operator-kernel-coverage.md) explains
  the coverage shape by operator family.
- [DSP/MVE Coverage](../guides/dsp-mve-coverage.md) explains why Ambiq focuses
  on MVE, DSP, and graph glue operators.
- This page documents how the generated API reference connects to the Sphinx
  site.

## Integration roadmap

| Phase | Goal | Notes |
|---|---|---|
| 1 | Build Sphinx pages from Doxygen XML. | Implemented with Breathe and Exhale. |
| 2 | Deploy the Sphinx artifact to Pages. | Release workflow publishes the Sphinx site after pack generation; docs-only Pages deploy remains gated. |
| 3 | Tune generated API grouping and titles. | Improve customer navigation through function groups and inherited CMSIS-NN material. |
| 4 | Generate coverage indexes from XML/source metadata. | Cross-link operator families to exact functions, files, dtypes, and acceleration paths. |
| 5 | Consider disabling standalone Doxygen HTML for website builds. | Keep it only where pack tooling still requires it. |

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
license terms. Keep generated API documentation attribution intact when
rendering or transforming Doxygen output.
