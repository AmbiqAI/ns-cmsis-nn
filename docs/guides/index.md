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
<strong>DSP/MVE Coverage</strong>
<em>Why MVE is a primary target where available, and where DSP remains important.</em>
</a>

<a class="doc-card" href="releases.html">
<span>Delivery</span>
<strong>Versioning & Releases</strong>
<em>Release tags, generated artifacts, release-please, and what ships together.</em>
</a>

<a class="doc-card" href="toolchains.html">
<span>Build contract</span>
<strong>Toolchain Pinning</strong>
<em>Compiler IDs, CPU flags, ABI checks, and what CMake validates for you.</em>
</a>

</div>

## Common Questions

| Question | Guide |
|---|---|
| Which operators and dtypes are covered? | [Operator & Kernel Coverage](operator-kernel-coverage.md) |
| Why emphasize MVE and still keep DSP paths? | [DSP/MVE Coverage](dsp-mve-coverage.md) |
| What exactly is in a release? | [Versioning & Releases](releases.md) |
| Which compiler and CPU flags were used? | [Toolchain Pinning](toolchains.md) |
