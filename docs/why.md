# Why heliaCORE?

heliaCORE is Ambiq's optimized neural-network kernel layer for Ambiq silicon.
It is implemented in this repository as `ns-cmsis-nn` and builds on the Arm
CMSIS-NN and CMSIS-Pack ecosystem with attribution, license preservation, and
CMSIS-compatible integration paths where applicable.

Arm CMSIS-NN provides the trusted, vendor-neutral foundation for efficient
neural-network kernels on Cortex-M. heliaCORE keeps that foundation visible and
usable while adding Ambiq delivery, integration, and kernel work around the
models Ambiq expects to run on Apollo-class devices.

## The problem it solves

Teams shipping production AI on Ambiq Cortex-M accelerator targets need more than a
collection of source files. They need a kernel layer that can move through a
firmware organization cleanly: pinned, packaged, checked at configure time, and
ready for both runtime and compiler-generated inference paths.

In HELIA workloads, the performance story is also broader than the largest
convolution or matrix-multiply layer. Field-like Ambiq model graphs spend real
time in padding, activations, reductions, reshapes, quantization, and other
operators around the obvious MAC-heavy blocks. When those surrounding operators
fall back to generic paths, end-to-end latency can still suffer even if the main
layers are highly optimized.

heliaCORE focuses on that full graph path:

- Optimized kernels that are **engineered for Ambiq silicon** and Ambiq field
  workloads.
- Coverage for both **MVE** and **DSP** paths where those acceleration features
  are available on Ambiq targets.
- Attention to **glue operators**, not only the largest arithmetic kernels.
- A **stable consumption contract**: versioned releases, checksummed artifacts,
  and reproducible packaging.
- Integrations for **CMake, Zephyr, CMSIS-Pack, and neuralSPOT-X** so downstream
  projects do not each invent their own glue.
- **Prebuilt artifacts** for common Cortex-M variants, with source builds still
  available where projects need them.
- Clear **toolchain pinning** for compiler, CPU/FPU flags, and ABI expectations.

## What changes for a firmware team

<div class="doc-card-grid" markdown>

<div class="doc-card" markdown>
<span>Integration</span>
<strong>One kernel package, several entry points</strong>
<em>Use the same release through CMake, CMSIS-Pack, Zephyr, or neuralSPOT-X instead of maintaining separate downstream copies.</em>
</div>

<div class="doc-card" markdown>
<span>Acceleration</span>
<strong>MVE and DSP stay part of the plan</strong>
<em>Cortex-M55 MVE paths are a primary target, while DSP variants remain important for Apollo-class devices without MVE.</em>
</div>

<div class="doc-card" markdown>
<span>Release control</span>
<strong>Artifacts that can be pinned</strong>
<em>Versioned releases, checksums, CMSIS-Pack metadata, and prebuilt archives give firmware teams something concrete to qualify.</em>
</div>

<div class="doc-card" markdown>
<span>HELIA fit</span>
<strong>Built for runtime and compiler paths</strong>
<em>heliaRT, heliaAOT, and neuralSPOT-X can all consume the same kernel layer instead of treating kernels as local project plumbing.</em>
</div>

</div>

heliaCORE is that foundation library for Ambiq AI deployments. It combines
CMSIS-NN-compatible operator surface area, Ambiq/HELIA kernel work, and
product-grade delivery:

| Layer        | What heliaCORE provides                                        |
|--------------|----------------------------------------------------------------|
| Kernels      | Ambiq-optimized neural-network operators for Ambiq Cortex-M accelerator paths. |
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
