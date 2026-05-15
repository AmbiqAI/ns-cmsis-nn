# Getting Started

heliaCORE supports four first-class consumption paths. Pick whichever
matches your existing build system:

<div class="heliacore-cards" markdown>

<a class="card" href="../cmake/" markdown>
### CMake `find_package`
Bring in a per-arch prebuilt tarball via a plain CMake project.
**Best for:** custom firmware projects already using CMake.
</a>

<a class="card" href="../cmsis-pack/" markdown>
### CMSIS-Pack
Install the `.pack` and pick **Source** or **Prebuilt** Cvariant.
**Best for:** Keil MDK, CMSIS-Toolbox, IAR users.
</a>

<a class="card" href="../zephyr/" markdown>
### Zephyr Module
`west`-managed module with Kconfig knobs.
**Best for:** Zephyr-based firmware.
</a>

<a class="card" href="../neuralspot-x/" markdown>
### neuralSPOT-X
Already wired through NSX's CMake graph.
**Best for:** anyone using NSX or heliaRT.
</a>

</div>

## Picking a release

Every heliaCORE release ships these assets to its [GitHub Release](https://github.com/AmbiqAI/ns-cmsis-nn/releases):

- **`ns-cmsis-nn-<cpu>-gcc-<version>.tar.gz`** — per-arch SDK tarball
  (sources + prebuilt `.a` + CMake config + manifest). Use with
  [CMake `find_package`](cmake.md).
- **`libns-cmsis-nn-<cpu>-<version>.a`** — bare static library if you
  only need the archive.
- **`Ambiq.NS-CMSIS-NN.<version>.pack`** — CMSIS-Pack. Use with
  [CMSIS-Pack tooling](cmsis-pack.md).
- **`*.sha256`** — SHA-256 checksums for every artifact.

All artifacts are built with **GCC 13.2 (GNU Arm Embedded)** for
**`cortex-m0+`**, **`cortex-m4`** (with FPv4 SP-D16), and
**`cortex-m55`** (with MVE).
