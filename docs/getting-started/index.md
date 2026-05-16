# Getting Started

Start with the integration path that matches the project you are building. Each
path consumes the same heliaCORE kernel release, but packages it for a different
tooling workflow.

```{toctree}
:hidden:
:maxdepth: 1

cmake
cmsis-pack
zephyr
neuralspot-x
```

## Choose Your Path

Use your project shape as the decision point: pick the path that owns dependency
resolution in your firmware today. Each path page gives you the exact download,
build setting, and verification step for that workflow.

<div class="path-choice-grid">

<a class="path-choice" href="cmake.html">
<span class="path-choice-label">You own a CMake firmware build</span>
<strong>Use the CMake package</strong>
<span>Download the per-CPU tarball, add it to your prefix path, and link the exported <code>nsx::cmsis_nn</code> target.</span>
<em>Open CMake guide</em>
</a>

<a class="path-choice" href="cmsis-pack.html">
<span class="path-choice-label">You use Keil, IAR, or CMSIS-Toolbox</span>
<strong>Install the CMSIS-Pack</strong>
<span>Let your IDE or CMSIS tooling select the heliaCORE component, then choose source or prebuilt Cvariant.</span>
<em>Open CMSIS-Pack guide</em>
</a>

<a class="path-choice" href="zephyr.html">
<span class="path-choice-label">You are building a Zephyr app</span>
<strong>Add the Zephyr module</strong>
<span>Add heliaCORE to <code>west.yml</code>, enable <code>CONFIG_NS_CMSIS_NN</code>, and let Zephyr compile or import it.</span>
<em>Open Zephyr guide</em>
</a>

<a class="path-choice" href="neuralspot-x.html">
<span class="path-choice-label">You are already in neuralSPOT-X</span>
<strong>Use the NSX integration</strong>
<span>NSX resolves the heliaCORE target for you and keeps board flags aligned with the rest of the HELIA stack.</span>
<em>Open NSX guide</em>
</a>

</div>

If you are integrating from scratch, start with the [CMake package](cmake.md).
If your tool already understands CMSIS components, start with the
[CMSIS-Pack](cmsis-pack.md). Zephyr and neuralSPOT-X users should follow their
native integration pages.

## What to Download

Every heliaCORE release publishes integration artifacts on the
[GitHub Release](https://github.com/AmbiqAI/ns-cmsis-nn/releases). Most users
download one integration artifact plus its matching checksum.

<div class="artifact-choice-grid">

<div class="artifact-choice artifact-choice-primary">
<span class="artifact-choice-label">Recommended for CMake</span>
<strong><code>ns-cmsis-nn-&lt;cpu&gt;-gcc-&lt;version&gt;.tar.gz</code></strong>
<span>The full SDK package: headers, source snapshot, prebuilt archive, CMake config, and manifest. This is the right starting point for custom CMake firmware.</span>
</div>

<div class="artifact-choice">
<span class="artifact-choice-label">Recommended for CMSIS tooling</span>
<strong><code>Ambiq.NS-CMSIS-NN.&lt;version&gt;.pack</code></strong>
<span>The CMSIS-Pack for IDE and CMSIS-Toolbox workflows. Use it when your project resolves dependencies through CMSIS components.</span>
</div>

<div class="artifact-choice">
<span class="artifact-choice-label">Advanced use</span>
<strong><code>libns-cmsis-nn-&lt;cpu&gt;-&lt;version&gt;.a</code></strong>
<span>The bare static library only. Choose this when your build already owns include paths, CPU flags, and link wiring.</span>
</div>

<div class="artifact-choice">
<span class="artifact-choice-label">Verification</span>
<strong><code>*.sha256</code></strong>
<span>Checksum files for validating downloads before you add them to a firmware build or internal package mirror.</span>
</div>

</div>

Choose the CPU name in the artifact that matches your target:

- **`cortex-m0`** for Cortex-M0/M0+ baseline Apollo targets.
- **`cortex-m4`** for Cortex-M4 targets with FPv4 SP-D16.
- **`cortex-m55`** for Cortex-M55 targets with MVE.

The `cortex-m0` artifact is the Cortex-M0/M0+ baseline package used for Apollo
targets in that class.

The prebuilt artifacts are built with **GCC 13.2 (GNU Arm Embedded)**. Source
integration paths, such as the CMSIS-Pack `Source` Cvariant or Zephyr source
mode, compile the kernels with your project toolchain instead.

For API and operator coverage, see [Operator & Kernel Coverage](../guides/operator-kernel-coverage.md)
and [API](../reference/api-groups.md).
