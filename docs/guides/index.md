# Guides

Use these guides when you need more detail than the integration quick starts.
They cover operator scope, acceleration paths, release mechanics, and toolchain
contracts.

<div class="doc-card-grid">

<a class="doc-card" href="operator-kernel-coverage.html">
<span>Coverage</span>
<strong>Operator & Kernel Coverage</strong>
<em>Operator families, data types, backend paths, and where to find exact C APIs.</em>
</a>

<a class="doc-card" href="dsp-mve-coverage.html">
<span>Acceleration</span>
<strong>Cortex-M Accelerators</strong>
<em>Scalar C, DSP, MVE/Helium, and floating-point paths for Ambiq Cortex-M kernels.</em>
</a>

<a class="doc-card" href="kernel-benchmarks.html">
<span>Performance</span>
<strong>Kernel Benchmarks</strong>
<em>Cycle-count charts comparing DSP and MVE paths on Apollo hardware.</em>
</a>

<a class="doc-card" href="releases.html">
<span>Delivery</span>
<strong>Versioning & Releases</strong>
<em>Release tags, generated artifacts, release-please, and what ships together.</em>
</a>

<a class="doc-card" href="toolchains.html">
<span>Build contract</span>
<strong>Toolchain Pinning</strong>
<em>ATfE recommendation, GCC prebuilt ABI checks, CPU flags, and CMake validation.</em>
</a>

</div>

## Common Questions

| Question | Guide |
|---|---|
| Which operators and dtypes are covered? | [Operator & Kernel Coverage](operator-kernel-coverage.md) |
| What are the Cortex-M accelerator paths? | [Cortex-M Accelerators](dsp-mve-coverage.md) |
| How fast is MVE vs DSP? | [Kernel Benchmarks](kernel-benchmarks.md) |
| What exactly is in a release? | [Versioning & Releases](releases.md) |
| Which compiler should I use? | [Toolchain Pinning](toolchains.md) |
