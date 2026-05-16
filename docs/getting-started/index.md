# Getting Started

Start with the integration path that matches the project you are building. Each
path consumes the same heliaCORE kernel release, but packages it for a different
tooling workflow.

```{toctree}
:hidden:
:maxdepth: 1

cmake
cmsis-pack
zephyr
neuralspot-x
```

<div class="heliacore-cards getting-started-cards">

<a class="card start-card" href="cmake.html">
<span class="start-card-label">Prebuilt SDK</span>
<strong>CMake package</strong>
<span>Use a per-architecture tarball with an exported CMake target.</span>
<em>Best for custom firmware projects already using CMake.</em>
</a>

<a class="card start-card" href="cmsis-pack.html">
<span class="start-card-label">CMSIS tooling</span>
<strong>CMSIS-Pack</strong>
<span>Install the pack and select a source or prebuilt C variant.</span>
<em>Best for Keil MDK, CMSIS-Toolbox, and IAR users.</em>
</a>

<a class="card start-card" href="zephyr.html">
<span class="start-card-label">RTOS module</span>
<strong>Zephyr Module</strong>
<span>Use a West-managed module with Kconfig options.</span>
<em>Best for Zephyr-based firmware.</em>
</a>

<a class="card start-card" href="neuralspot-x.html">
<span class="start-card-label">HELIA stack</span>
<strong>neuralSPOT-X</strong>
<span>Consume heliaCORE through the NSX CMake graph.</span>
<em>Best for NSX or heliaRT applications.</em>
</a>

</div>

## Choose Your Path

| If you have... | Start with... |
|---|---|
| A plain CMake firmware project | [CMake package](cmake.md) |
| CMSIS tooling, Keil, IAR, or CMSIS-Toolbox | [CMSIS-Pack](cmsis-pack.md) |
| A Zephyr application | [Zephyr Module](zephyr.md) |
| A HELIA/neuralSPOT-X application | [neuralSPOT-X](neuralspot-x.md) |

## Release Artifacts

Every heliaCORE release ships these assets to its [GitHub Release](https://github.com/AmbiqAI/ns-cmsis-nn/releases):

| Artifact | Use it for |
|---|---|
| `ns-cmsis-nn-<cpu>-gcc-<version>.tar.gz` | CMake package with sources, prebuilt archive, config, and manifest. |
| `libns-cmsis-nn-<cpu>-<version>.a` | Bare static archive when your build already owns integration. |
| `Ambiq.NS-CMSIS-NN.<version>.pack` | CMSIS-Pack workflows and IDE/tooling integration. |
| `*.sha256` | Checksum verification for release artifacts. |

All artifacts are built with **GCC 13.2 (GNU Arm Embedded)** for
**`cortex-m0+`**, **`cortex-m4`** (with FPv4 SP-D16), and
**`cortex-m55`** (with MVE).

:::{note} Artifact CPU names
Release artifact names use `cortex-m0`, `cortex-m4`, and `cortex-m55`. The
`cortex-m0` artifact is the Cortex-M0/M0+ baseline package used for Apollo
targets in that class.
:::

For API and operator coverage, see [Operator & Kernel Coverage](../guides/operator-kernel-coverage.md)
and [Doxygen API](../reference/doxygen.md).
