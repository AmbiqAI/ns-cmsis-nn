---
title: heliaCORE
hide:
  - navigation
  - toc
---

<section class="landing-hero" markdown>

<div class="hero-copy" markdown>

![heliaCORE](assets/helia-core-logo-dark.png){ .hero-logo .hero-logo--core }

<p class="hero-kicker"><span></span>Optimized AI kernels for Apollo silicon</p>

# Kernel acceleration for Ambiq AI.

**heliaCORE** is Ambiq's optimized neural-network kernel library for Ambiq
silicon, implemented here as `ns-cmsis-nn`. It builds on the Arm CMSIS-NN and
CMSIS-Pack ecosystem, then adds Ambiq-tuned operators and HELIA integration
paths for Apollo-class Cortex-M DSP/MVE targets, with particular focus on MVE
acceleration where supported.

<div class="hero-actions" markdown>
[Get started :material-arrow-right:](getting-started/index.md){ .md-button .md-button--primary }
[Why heliaCORE](why.md){ .md-button }
</div>

</div>

<div class="hero-panel hero-panel--core" markdown>
<div class="panel-topline"><span>HELIA CORE</span><span class="status-live">V7.25</span></div>
<div class="metric-grid" markdown>
<div><strong>200+</strong><span>MVE/DSP ops</span></div>
<div><strong>40+</strong><span>Field models</span></div>
<div><strong>53</strong><span>Op types</span></div>
<div><strong>4</strong><span>Paths</span></div>
</div>

```cmake
find_package(ns-cmsis-nn REQUIRED CONFIG)
target_link_libraries(app PRIVATE nsx::cmsis_nn)
```

</div>

</section>

<div class="logo-rail logo-rail--core" markdown>
<span>CMSIS-Pack</span>
<span>Zephyr</span>
<span>CMake</span>
<span>neuralSPOT-X</span>
<span>GCC</span>
<span>Cortex-M0+</span>
<span>Cortex-M4</span>
<span>Cortex-M55</span>
<span>Apollo510</span>
</div>

<section class="section-block section-block--intro" markdown>

<p class="section-eyebrow">What it is</p>

## Ambiq's foundation neural-network kernel layer.

heliaCORE is the productized form of `ns-cmsis-nn`: a production kernel layer
for Ambiq AI workloads. It preserves compatibility with inherited CMSIS-NN APIs
where that surface applies, adds Ambiq-optimized kernels for Ambiq silicon, and
packages the result for HELIA workflows.

</section>

<section class="section-block" markdown>

<p class="section-eyebrow">Ambiq workload coverage</p>

## Extending coverage for Ambiq field-like models.

Arm CMSIS-NN provides the trusted foundation for efficient neural-network
kernels on Cortex-M. In Ambiq's HELIA workflows, internal profiling across a
field-like model suite also highlighted important Ambiq-specific coverage needs:
real models spend measurable time in PAD, LeakyReLU, and other glue operators
that are easy to overlook when focusing only on the largest MAC-heavy layers.

heliaCORE responds to those Ambiq silicon needs by adding 200+ DSP/MVE optimized
operators and extra variants around the CMSIS-NN-compatible surface.

<div class="workflow workflow--accel" markdown>

<div class="workflow-step" markdown>
<span>1</span>
<strong>MVE-first where available</strong>
<p>Cortex-M55 paths are a primary optimization target, with vectorized kernels
for Ambiq workloads where MVE can move latency.</p>
</div>

<div class="workflow-step" markdown>
<span>2</span>
<strong>DSP coverage for Apollo-class MCUs</strong>
<p>Cortex-M DSP paths remain important for Apollo targets that do not have MVE,
so the library keeps DSP-optimized variants in the release surface.</p>
</div>

<div class="workflow-step" markdown>
<span>3</span>
<strong>Glue operators count too</strong>
<p>Coverage extends beyond obvious MAC-heavy layers to operators that shape real
model latency in HELIA deployments.</p>
</div>

</div>

<div class="takeaway-grid" markdown>

<div class="takeaway-card" markdown>
<span class="card-icon">A</span>
<strong>Ambiq field-like suite</strong>
<span>40+ models, 53 operator types, 247 unique operators, and 963 operator instances.</span>
</div>

<div class="takeaway-card" markdown>
<span class="card-icon">M</span>
<strong>MLPerf Tiny baseline</strong>
<span>5 models, 7 operator types, 34 unique operators, and 80 operator instances.</span>
</div>

<div class="takeaway-card" markdown>
<span class="card-icon">+</span>
<strong>Expanded coverage</strong>
<span>200+ DSP/MVE optimized operators for Ambiq silicon, plus additional variants for existing operators.</span>
</div>

</div>

[Explore DSP/MVE coverage data →](guides/dsp-mve-coverage.md){ .text-link }

[Browse operator & kernel coverage →](guides/operator-kernel-coverage.md){ .text-link }

</section>

<section class="section-block" markdown>

<p class="section-eyebrow">Pick your integration</p>

## Four supported ways in.

<div class="platform-grid" markdown>

<a class="platform-card platform-card--cmsis" href="getting-started/cmake/" markdown>
<span class="logo-mark">CM</span>
<span class="platform-label">Prebuilt SDK</span>
<strong>CMake `find_package`</strong>
<span>Drop in a per-architecture tarball and link `nsx::cmsis_nn`. CPU and compiler are validated at configure time.</span>
<em>Start with CMake →</em>
</a>

<a class="platform-card" href="getting-started/cmsis-pack/" markdown>
<span class="logo-mark logo-mark--amber">PK</span>
<span class="platform-label">CMSIS tooling</span>
<strong>CMSIS-Pack</strong>
<span>Install `Ambiq.NS-CMSIS-NN.<version>.pack` and choose **Source** or **Prebuilt** Cvariant.</span>
<em>Use the pack →</em>
</a>

<a class="platform-card" href="getting-started/zephyr/" markdown>
<span class="logo-mark logo-mark--yellow">Z</span>
<span class="platform-label">RTOS module</span>
<strong>Zephyr Module</strong>
<span>West-managed module with `CONFIG_NS_CMSIS_NN` and an optional prebuilt archive path.</span>
<em>Wire Zephyr →</em>
</a>

<a class="platform-card" href="getting-started/neuralspot-x/" markdown>
<span class="logo-mark logo-mark--brown">NSX</span>
<span class="platform-label">HELIA stack</span>
<strong>neuralSPOT-X</strong>
<span>NSX consumes heliaCORE through its CMake graph and exposes the kernels to heliaRT-based applications.</span>
<em>Use NSX →</em>
</a>

</div>

</section>

<section class="section-block" markdown>

<p class="section-eyebrow">HELIA family</p>

## Powering heliaRT and heliaAOT.

heliaCORE provides the optimized kernel layer that powers Ambiq's edge-AI
runtime and compiler flows. If you're looking for the runtime + tooling that
uses these kernels, see
[**heliaRT :material-open-in-new:**](https://ambiqai.github.io/helia-rt/){target="_blank"}
— Ambiq's optimized LiteRT runtime for Apollo.

heliaAOT is the ahead-of-time compiler path for deterministic, low-footprint
firmware. heliaRT and heliaAOT both rely on optimized HELIA kernels where
applicable; heliaCORE is the shared kernel foundation behind those paths.

!!! info "Arm CMSIS ecosystem"
    heliaCORE is built on and distributed for the Arm CMSIS ecosystem, including
    CMSIS-NN-compatible APIs and CMSIS-Pack delivery. Ambiq-specific additions
    are intended to ease integration into the HELIA AI platform for Ambiq
    silicon. For vendor-neutral Cortex-M kernel work, Arm CMSIS-NN remains the
    upstream ecosystem reference.

</section>
