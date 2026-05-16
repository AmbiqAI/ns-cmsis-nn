# Why heliaCORE?

heliaCORE is Ambiq's optimized neural-network kernel layer for Ambiq silicon.
It is implemented in this repository as `ns-cmsis-nn` and builds on the Arm
CMSIS-NN and CMSIS-Pack ecosystem with attribution, license preservation, and
CMSIS-compatible integration paths where applicable.

Teams shipping production AI on Ambiq Cortex-M DSP/MVE targets need a kernel
library that is easy to integrate into HELIA workflows:

- Optimized kernels that are **engineered for Ambiq silicon** and Ambiq field
  workloads.
- A **stable consumption contract** (versioned, checksummed, reproducible).
- A way to **plug into multiple build systems** (CMake, Zephyr, CMSIS-Pack,
  neuralSPOT-X) without forking glue into each downstream repo.
- **Prebuilt** artifacts for the common Cortex-M variants so most users
  never recompile the kernels themselves.
- A clear story on **toolchain pinning** — the version of GCC, the
  `-mcpu`/`-mfpu` flags, and the ABI baked into a `.a`.

heliaCORE is that foundation library for Ambiq AI deployments. It combines
CMSIS-NN-compatible operator surface area, Ambiq/HELIA kernel work, and
product-grade delivery:

| Layer        | What heliaCORE provides                                        |
|--------------|----------------------------------------------------------------|
| Kernels      | Ambiq-optimized neural-network operators for Ambiq DSP/MVE targets. |
| Compatibility| CMSIS-NN-style `arm_*` APIs where inherited and supported.      |
| Packaging    | Per-arch `.a` + SDK tarball + CMSIS-Pack + Zephyr module.      |
| Distribution | release-please-managed `vX.Y.Z` tags + GitHub Releases.        |
| Validation   | `find_package` configure-time `-mcpu` and compiler-ID checks.  |
| Integration  | First-class neuralSPOT-X target, Zephyr `west` integration.    |

Use Arm's upstream [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) directly
when you want the vendor-neutral CMSIS-NN project. Use heliaCORE when you are
building for Ambiq silicon and want the kernel layer Ambiq integrates into the
HELIA AI stack, with artifacts you can pin and ship.

## Attribution and license posture

heliaCORE depends on the strength of the Arm CMSIS ecosystem. The inherited
CMSIS-NN source and API documentation remain under their original license terms,
including Apache-2.0 where applicable. Ambiq-specific packaging, integrations,
and additions are documented separately and are intended to make CMSIS and HELIA
workflows easier on Ambiq devices.

## What stays compatible

- The inherited CMSIS-NN-style `arm_*` entry points, signatures, and behavior
  where those APIs are supported.
- Doxygen comments for inherited APIs, kept close to upstream when possible.
- License notices and attribution for inherited Arm CMSIS-NN sources.

## What's specific to heliaCORE

- Additional Ambiq/HELIA kernel implementations and tuning for Ambiq devices.
- The build system (`Source/CMakeLists.txt`, toolchain files, find_package
  config), the pack manifest (`Ambiq.NS-CMSIS-NN.pdsc`), and the Zephyr
  module.
- The release pipeline (`.github/workflows/release.yml`) and artifact
  layout under each GitHub Release.
- The integration glue with [neuralSPOT-X](https://github.com/AmbiqAI/neuralSPOT)
  [heliaRT](https://github.com/AmbiqAI/helia-rt), and HELIA compiler
  flows such as heliaAOT.

## Where it fits

heliaCORE is part of the **foundation layer** of the HELIA AI platform.
It sits below runtime and compiler paths:

- **heliaRT** is the runtime path. heliaCORE is the optimized kernel library
  underneath it.
- **heliaAOT** is the ahead-of-time compiler path. heliaCORE provides optimized
  kernels that make generated inference code efficient.
- **neuralSPOT-X / NSX** is the SDK and integration layer. NSX-based projects
  can consume heliaCORE through CMake, Zephyr, or release artifacts.
