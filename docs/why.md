# Why heliaCORE?

Arm's [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) is a great kernel
library. It is also delivered as a source tree that you bring into your build,
period. Teams shipping production firmware on Apollo silicon need more than
that:

- A **stable consumption contract** (versioned, signed, reproducible).
- A way to **plug into multiple build systems** (CMake, Zephyr, CMSIS-Pack,
  neuralSPOT-X) without forking glue into each downstream repo.
- **Prebuilt** artifacts for the common Cortex-M variants so most users
  never recompile the kernels themselves.
- A clear story on **toolchain pinning** — the version of GCC, the
  `-mcpu`/`-mfpu` flags, and the ABI baked into a `.a`.

heliaCORE is that wrapper. It tracks upstream CMSIS-NN and adds:

| Layer        | What heliaCORE adds                                            |
|--------------|----------------------------------------------------------------|
| Source       | Unmodified upstream kernels + Apollo-friendly CMake graph.     |
| Packaging    | Per-arch `.a` + SDK tarball + CMSIS-Pack + Zephyr module.      |
| Distribution | release-please-managed `vX.Y.Z` tags + GitHub Releases.        |
| Validation   | `find_package` configure-time `-mcpu` and compiler-ID checks.  |
| Integration  | First-class neuralSPOT-X target, Zephyr `west` integration.    |

If the only thing you want is *the latest upstream sources*, use
ARM-software/CMSIS-NN directly. If you want a **product** you can pin a
version of and ship, use heliaCORE.

## What stays upstream-aligned

- All `arm_*` public functions, their signatures, and their behavior.
- The Doxygen comments on those functions.
- The Apache-2.0 license on the kernel sources.

## What's specific to heliaCORE

- The build system (`Source/CMakeLists.txt`, toolchain files, find_package
  config), the pack manifest (`Ambiq.NS-CMSIS-NN.pdsc`), and the Zephyr
  module.
- The release pipeline (`.github/workflows/release.yml`) and artifact
  layout under each GitHub Release.
- The integration glue with [neuralSPOT-X](https://github.com/AmbiqAI/neuralSPOT)
  and [heliaRT](https://github.com/AmbiqAI/helia-rt).
