# heliaCORE

<div class="landing-hero">
  <div>
    <img class="hero-logo" src="_static/helia-core-logo-dark.png" alt="heliaCORE">
    <p class="hero-kicker">Optimized AI kernels for Apollo silicon</p>
    <h1 class="hero-title">Kernel acceleration for Ambiq AI.</h1>
    <p class="hero-badge">ATfE + MVE ready</p>
    <p class="hero-summary"><strong>heliaCORE</strong> is Ambiq's optimized neural-network kernel library for Ambiq silicon. This repository delivers it as <code>ns-cmsis-nn</code>: a CMSIS-NN-based kernel layer with Ambiq-tuned operators, CMSIS-Pack delivery, and HELIA integration paths for Apollo-class Cortex-M accelerators.</p>
    <div class="hero-actions">
      <a class="hc-button hc-button-primary" href="getting-started/index.html">Get started</a>
      <a class="hc-button" href="why.html">Why heliaCORE</a>
      <a class="hc-button" href="reference/api-groups.html">API</a>
    </div>
  </div>
  <div class="hero-panel">
    <div class="panel-topline"><span>HELIA CORE</span><span class="status-live">V7.25</span></div>
    <div class="metric-grid">
      <div><strong>200+</strong><span>accelerated ops</span></div>
      <div><strong>40+</strong><span>Field models</span></div>
      <div><strong>53</strong><span>Op types</span></div>
      <div><strong>4</strong><span>Paths</span></div>
    </div>

```cmake
find_package(ns-cmsis-nn REQUIRED CONFIG)
target_link_libraries(app PRIVATE nsx::cmsis_nn)
```
  </div>
</div>

<div class="logo-rail">
  <span>CMSIS-Pack</span>
  <span>Zephyr</span>
  <span>CMake</span>
  <span>neuralSPOT-X</span>
  <span>ATfE</span>
  <span>GCC</span>
  <span>Cortex-M0+</span>
  <span>Cortex-M4</span>
  <span>Cortex-M55</span>
  <span>Apollo510</span>
</div>

```{toctree}
:hidden:
:maxdepth: 2

why
getting-started/index
guides/index
guides/operator-kernel-coverage
guides/dsp-mve-coverage
guides/releases
guides/toolchains
reference/api-groups
contributing
```

<section class="section-block">
<p class="section-eyebrow">What it is</p>

## A kernel layer for HELIA AI workloads

heliaCORE is Ambiq's optimized neural-network kernel layer for Apollo-class
devices. In this repository, it is delivered as `ns-cmsis-nn`: inherited
CMSIS-NN-compatible APIs where they apply, Ambiq-tuned kernels where HELIA needs
more coverage, and packaging for runtimes, compilers, and firmware builds.
</section>

<section class="section-block">
<p class="section-eyebrow">Ambiq workload coverage</p>

## Coverage for real Ambiq model graphs

Arm CMSIS-NN provides the trusted foundation for efficient neural-network
kernels on Cortex-M. In Ambiq's HELIA workflows, internal profiling across
field-like models also highlighted important Ambiq-specific coverage needs:
real graphs spend measurable time in padding, activations, reductions, and
other operators around the largest MAC-heavy layers.

heliaCORE broadens Cortex-M accelerator coverage around the CMSIS-NN-compatible
foundation so those end-to-end paths stay optimized on Ambiq silicon.

<div class="workflow">
  <div class="workflow-step">
    <span class="workflow-icon">MVE</span>
    <strong>MVE-first where available</strong>
    <p>Cortex-M55 paths are a primary optimization target, with vectorized kernels where MVE can improve end-to-end latency.</p>
  </div>
  <div class="workflow-step">
    <span class="workflow-icon">DSP</span>
    <strong>DSP coverage for Apollo-class MCUs</strong>
    <p>Cortex-M DSP paths remain important on Apollo targets without MVE, so DSP variants stay part of the release surface.</p>
  </div>
  <div class="workflow-step">
    <span class="workflow-icon">Flow</span>
    <strong>Glue operators count too</strong>
    <p>Coverage extends beyond obvious MAC-heavy layers to the graph operators that shape real HELIA deployments.</p>
  </div>
</div>

<div class="takeaway-grid">
  <div class="takeaway-card">
    <span class="card-icon">40+</span>
    <strong>Ambiq field-like suite</strong>
    <span>40+ models, 53 operator types, 247 unique operators, and 963 operator instances.</span>
  </div>
  <div class="takeaway-card">
    <span class="card-icon">Tiny</span>
    <strong>MLPerf Tiny baseline</strong>
    <span>5 models, 7 operator types, 34 unique operators, and 80 operator instances.</span>
  </div>
  <div class="takeaway-card">
    <span class="card-icon">200+</span>
    <strong>Expanded coverage</strong>
    <span>200+ accelerated operators for Ambiq silicon, plus additional variants for existing operators.</span>
  </div>
</div>

[Explore Cortex-M accelerators](guides/dsp-mve-coverage.md)
[Browse operator & kernel coverage](guides/operator-kernel-coverage.md)
</section>

<section class="section-block">
<p class="section-eyebrow">Pick your integration</p>

## Four supported ways in

<div class="platform-grid">
  <a class="platform-card" href="getting-started/cmake.html">
    <span class="platform-label">Prebuilt SDK</span>
    <strong>CMake package</strong>
    <span>Link a per-architecture package with the exported CMake target; configure validates CPU and compiler.</span>
    <em>Start with CMake</em>
  </a>
  <a class="platform-card" href="getting-started/cmsis-pack.html">
    <span class="platform-label">CMSIS tooling</span>
    <strong>CMSIS-Pack</strong>
    <span>Install the Ambiq pack and select a source or prebuilt variant for your project.</span>
    <em>Use the pack</em>
  </a>
  <a class="platform-card" href="getting-started/zephyr.html">
    <span class="platform-label">RTOS module</span>
    <strong>Zephyr Module</strong>
    <span>Enable the module with Kconfig and optionally point it at a prebuilt archive.</span>
    <em>Wire Zephyr</em>
  </a>
  <a class="platform-card" href="getting-started/neuralspot-x.html">
    <span class="platform-label">HELIA stack</span>
    <strong>neuralSPOT-X</strong>
    <span>Consume heliaCORE through the NSX CMake graph for heliaRT-based applications.</span>
    <em>Use NSX</em>
  </a>
</div>
</section>

<section class="section-block">
<p class="section-eyebrow">HELIA family</p>

## Powering heliaRT and heliaAOT

heliaCORE provides the optimized kernel layer that powers Ambiq's edge-AI
runtime and compiler flows. heliaRT and heliaAOT both rely on optimized HELIA
kernels where applicable; heliaCORE is the shared kernel foundation behind
those paths.

<div class="helia-family-grid">
  <a class="helia-family-card" href="https://ambiqai.github.io/helia-rt/">
    <span class="family-card-kicker">Runtime path</span>
    <strong>heliaRT</strong>
    <span>Ambiq's optimized LiteRT runtime for Apollo applications, powered by HELIA kernels from heliaCORE where they apply.</span>
    <em>Open heliaRT</em>
  </a>
  <a class="helia-family-card" href="why.html#where-it-fits">
    <span class="family-card-kicker">Compiler path</span>
    <strong>heliaAOT</strong>
    <span>Ahead-of-time generation for deterministic, low-footprint firmware, using heliaCORE kernels as the shared foundation.</span>
    <em>See where it fits</em>
  </a>
</div>

:::{important} Arm CMSIS ecosystem
heliaCORE is built on and distributed for the Arm CMSIS ecosystem, including
CMSIS-NN-compatible APIs and CMSIS-Pack delivery. Ambiq-specific additions are
intended to ease integration into the HELIA AI platform for Ambiq silicon. For
vendor-neutral Cortex-M kernel work, Arm CMSIS-NN remains the upstream ecosystem
reference.
:::
</section>
