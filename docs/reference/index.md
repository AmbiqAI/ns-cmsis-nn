# Reference

Reference material for the heliaCORE C API, generated API documentation, and
operator coverage summaries.

<div class="doc-card-grid">

<a class="doc-card" href="doxygen/">
<span>API docs</span>
<strong>Doxygen API</strong>
<em>Where the generated C API reference comes from, what it includes, and how HTML/XML output is used.</em>
</a>

<a class="doc-card" href="../guides/operator-kernel-coverage/">
<span>Coverage</span>
<strong>Operator & Kernel Coverage</strong>
<em>Concise operator-family, dtype, and backend overview before you dive into function signatures.</em>
</a>

<a class="doc-card" href="https://arm-software.github.io/CMSIS-NN/latest/">
<span>Upstream</span>
<strong>Arm CMSIS-NN Reference</strong>
<em>Vendor-neutral upstream API reference for inherited CMSIS-NN concepts and APIs.</em>
</a>

</div>

## API Source Of Truth

| Need | Start here |
|---|---|
| Exact function signatures and Doxygen comments | [Doxygen API](doxygen.md) |
| High-level operator and backend coverage | [Operator & Kernel Coverage](../guides/operator-kernel-coverage.md) |
| Inherited CMSIS-NN behavior and upstream context | [Arm CMSIS-NN Reference](https://arm-software.github.io/CMSIS-NN/latest/) |

The full heliaCORE reference will be generated from this repository's headers so
Ambiq-specific kernels and integration APIs are documented alongside inherited
CMSIS-NN-compatible functions. See [Why heliaCORE](../why.md#what-stays-compatible)
for the compatibility and attribution posture.
