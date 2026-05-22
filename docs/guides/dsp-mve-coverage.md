# Cortex-M Accelerators

heliaCORE maps CMSIS-NN-compatible operators onto the accelerator features
available on Ambiq Cortex-M targets.

In this guide, **Cortex-M accelerators** means:

- scalar C baseline kernels,
- the Cortex-M DSP extension,
- MVE/Helium vector instructions,
- and FPU/FP16 support where a kernel and device can use it.

MVE is the primary high-throughput path on Cortex-M55-class Apollo devices. DSP
paths remain important for Apollo-class devices that do not include MVE.

For a concise family-by-family summary, see
[Operator & Kernel Coverage](operator-kernel-coverage.md). For exact function
prototypes and per-kernel behavior, see [API](../reference/api-groups.md).

## Why coverage matters

Arm CMSIS-NN provides the trusted foundation for efficient neural-network kernels
on Cortex-M. Ambiq HELIA profiling showed that real model graphs spend time in
more than the largest convolution or matrix-multiply layer.

The practical gaps are often around:

- padding and layout changes,
- LeakyReLU and other activations,
- reductions and comparisons,
- quantization and classifier-tail operators,
- and wrapper/buffer helpers needed by firmware integration.

heliaCORE adds 200+ accelerated operators and variants so these graph edges do
not fall back to generic paths on Ambiq firmware targets.

## What the acceleration paths are

<div class="takeaway-grid" markdown>

<div class="takeaway-card" markdown>
<span class="api-tag">baseline</span>
<strong>Scalar C</strong>
<span>Portable kernels for compatibility and for smaller Cortex-M0-class devices without specialized math extensions.</span>
</div>

<div class="takeaway-card" markdown>
<span class="api-tag">fixed-point</span>
<strong>DSP extension</strong>
<span>Packed scalar instructions selected with `ARM_MATH_DSP` for efficient int8, int16, and int32 quantized kernels.</span>
</div>

<div class="takeaway-card" markdown>
<span class="api-tag">vector</span>
<strong>MVE / Helium</strong>
<span>Arm M-Profile Vector Extension, selected with MVE feature macros such as `ARM_MATH_MVEI`, for high-throughput Cortex-M55 kernels.</span>
</div>

<div class="takeaway-card" markdown>
<span class="api-tag">floating point</span>
<strong>FPU / FP16</strong>
<span>Hardware and compiler support for fp16/fp32 variants. It complements DSP or MVE rather than replacing them.</span>
</div>

</div>

## Why MVE is a big step

<div class="workflow workflow--accel" markdown>

<div class="workflow-step" markdown>
<span class="workflow-icon">MVE</span>
<strong>128-bit vector engine</strong>
<p>MVE, also known as Helium, lets Cortex-M55 process many narrow values at once:
16 int8 lanes, 8 int16 or fp16 lanes, or 4 int32 or fp32 lanes per vector.</p>
</div>

<div class="workflow-step" markdown>
<span class="workflow-icon">DSP</span>
<strong>Packed scalar math</strong>
<p>The DSP extension accelerates fixed-point math with packed 8-bit and 16-bit
operations, dual 16-bit MAC instructions, saturation, and fast accumulates.</p>
</div>

<div class="workflow-step" markdown>
<span class="workflow-icon">Flow</span>
<strong>Whole-graph speed</strong>
<p>Good kernels keep lanes busy across convolution, fully connected, activation,
pooling, quantization, and layout operators, not only the obvious MAC-heavy
layers.</p>
</div>

</div>

MVE is not just a different intrinsic name. It changes the unit of work:

- **DSP** improves scalar fixed-point loops with packed instructions.
- **MVE** runs 128-bit vector operations with predication, vector MAC,
  dot-product patterns, saturation, and lane-wise data movement.
- **Expected gain** is kernel and shape dependent, but favorable MAC-heavy int8
  paths can make an up-to-8x speedup over DSP-style implementations plausible
  when data layout and memory traffic cooperate.

That is why heliaCORE treats MVE coverage as a primary design axis for
Cortex-M55-class Ambiq devices, while still preserving DSP paths for Apollo
devices that rely on them.

## Coverage pressure from real model graphs

These internal counts explain why heliaCORE invests in both MAC-heavy kernels
and the smaller operators that connect them.

<div class="takeaway-grid" markdown>

<div class="takeaway-card" markdown>
<span class="api-tag">Ambiq field-like suite</span>
<strong>53 operator types</strong>
<span>247 unique operator configurations and 963 total operator instances across internal HELIA-oriented model graphs.</span>
</div>

<div class="takeaway-card" markdown>
<span class="api-tag">MLPerf Tiny baseline</span>
<strong>7 operator types</strong>
<span>34 unique operator configurations and 80 total instances in a familiar public benchmark suite used here only for scale.</span>
</div>

<div class="takeaway-card" markdown>
<span class="api-tag">Coverage target</span>
<strong>200+ optimized operators</strong>
<span>Accelerated variants cover both the high-cost math kernels and the glue operators that can dominate real deployment latency.</span>
</div>

</div>

How to read the counts:

- **Operator types** are distinct operator families.
- **Unique operators** include dtype, shape pattern, or variant differences.
- **Operator instances** are total appearances across model graphs.

MLPerf Tiny is included only as a familiar public scale reference.

:::{note} Scope
This coverage view is based on Ambiq's internal model suite. It is
not a performance benchmark and is not a statement about the broader Arm
CMSIS-NN ecosystem. For vendor-neutral Cortex-M kernel work, Arm CMSIS-NN
remains the upstream ecosystem reference.
:::

For measured cycle counts comparing MVE, DSP, and Scalar C on Apollo hardware, see
[Kernel Benchmarks](kernel-benchmarks.md).
