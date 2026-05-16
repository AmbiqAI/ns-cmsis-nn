# Getting Started

heliaCORE supports four first-class consumption paths. Pick whichever
matches your existing build system:

<div class="heliacore-cards getting-started-cards">

<a class="card start-card" href="cmake/">
<span class="start-card-label">Prebuilt SDK</span>
<strong>CMake package</strong>
<span>Bring in a per-architecture tarball with an exported CMake target.</span>
<em>Best for custom firmware projects already using CMake.</em>
</a>

<a class="card start-card" href="cmsis-pack/">
<span class="start-card-label">CMSIS tooling</span>
<strong>CMSIS-Pack</strong>
<span>Install the pack and select a source or prebuilt C variant.</span>
<em>Best for Keil MDK, CMSIS-Toolbox, and IAR users.</em>
</a>

<a class="card start-card" href="zephyr/">
<span class="start-card-label">RTOS module</span>
<strong>Zephyr Module</strong>
<span>Use a West-managed module with Kconfig options.</span>
<em>Best for Zephyr-based firmware.</em>
</a>

<a class="card start-card" href="neuralspot-x/">
<span class="start-card-label">HELIA stack</span>
<strong>neuralSPOT-X</strong>
<span>Consume heliaCORE through the NSX CMake graph.</span>
<em>Best for NSX or heliaRT applications.</em>
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
