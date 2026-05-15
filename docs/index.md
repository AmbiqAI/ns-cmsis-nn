---
title: heliaCORE
hide:
  - navigation
  - toc
---

<section class="heliacore-hero" markdown>

<div markdown>

# CMSIS-NN, packaged for Apollo.

<p class="lede" markdown>
**heliaCORE** is Ambiq's distribution of Arm CMSIS-NN, tuned for
Apollo silicon and shipped the way embedded teams actually consume
libraries: a CMSIS-Pack, a Zephyr module, a CMake `find_package`,
and a first-class neuralSPOT-X integration.
</p>

<div class="cta" markdown>
[Get started :material-arrow-right:](getting-started/index.md){ .md-button .md-button--primary }
[Why heliaCORE](why.md){ .md-button }
</div>

</div>

<div class="heliacore-stats" markdown>
<div class="tile"><span class="num">4</span><span class="label">Consumption Paths</span></div>
<div class="tile"><span class="num">3</span><span class="label">Cortex-M Targets</span></div>
<div class="tile"><span class="num">100%</span><span class="label">Upstream API</span></div>
<div class="tile"><span class="num">Apache&nbsp;2.0</span><span class="label">License</span></div>
</div>

</section>

<div class="heliacore-badges" markdown>
<span class="badge">CMSIS-Pack</span>
<span class="badge">Zephyr</span>
<span class="badge">CMake</span>
<span class="badge">neuralSPOT-X</span>
<span class="badge">GCC</span>
<span class="badge">Cortex-M0+</span>
<span class="badge">Cortex-M4</span>
<span class="badge">Cortex-M55</span>
<span class="badge">Apollo510</span>
</div>

## Pick your integration

<div class="heliacore-cards" markdown>

<a class="card" href="getting-started/cmake/" markdown>
### CMake `find_package`
Drop in a prebuilt per-architecture tarball and link `nsx::cmsis_nn`.
Toolchain and CPU are validated at configure time.
</a>

<a class="card" href="getting-started/cmsis-pack/" markdown>
### CMSIS-Pack
Install `Ambiq.NS-CMSIS-NN.<version>.pack` and select the **Source**
or **Prebuilt** Cvariant from your IDE / CMSIS-Toolbox project.
</a>

<a class="card" href="getting-started/zephyr/" markdown>
### Zephyr Module
West-managed module with `CONFIG_NS_CMSIS_NN`, plus an optional
`CONFIG_NS_CMSIS_NN_USE_PREBUILT` path that links the released `.a`.
</a>

<a class="card" href="getting-started/neuralspot-x/" markdown>
### neuralSPOT-X
First-class consumer. NSX wires heliaCORE through its CMake graph and
exposes the kernels to user models with zero extra glue.
</a>

</div>

## Part of the HELIA family

heliaCORE is the low-level kernel library underneath Ambiq's edge-AI stack.
If you're looking for the runtime + tooling that *uses* these kernels, see
[**heliaRT :material-open-in-new:**](https://ambiqai.github.io/helia-rt/){target="_blank"}
— Ambiq's optimized LiteRT runtime for Apollo.

!!! info "Upstream alignment"
    heliaCORE keeps the public CMSIS-NN API (`arm_*` functions) 1:1 with
    [ARM-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN). The
    fork adds packaging, Apollo-tuned defaults, prebuilt artifacts, and
    integration recipes — it does **not** alter the kernel ABIs.
