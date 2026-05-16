# Cortex-M Accelerators

heliaCORE's accelerator work focuses on Ambiq silicon. In this guide,
"Cortex-M accelerators" means the kernel paths selected by the target CPU and
compiler feature macros: scalar C baseline code, DSP instructions, MVE/Helium,
and floating-point support where a kernel and device can use it. MVE is a
primary target where it is available, especially for Cortex-M55-class Apollo
devices, while DSP optimized paths remain important for Apollo-class devices
that do not include MVE.

This page summarizes why heliaCORE includes Ambiq-specific operator additions
around the CMSIS-NN-compatible surface. The goal is integration into HELIA AI
workflows for Ambiq devices, with continued respect for the Arm CMSIS ecosystem
and CMSIS-Pack delivery.

For a concise family-by-family summary, see
[Operator & Kernel Coverage](operator-kernel-coverage.md). For exact function
prototypes and per-kernel behavior, see [API](../reference/api-groups.md).

## Why coverage matters

Arm CMSIS-NN provides the trusted foundation for efficient neural-network kernels
on Cortex-M. In Ambiq's HELIA workflows, internal profiling across a field-like
model suite highlighted additional Ambiq-specific coverage needs: real models can
spend measurable time in PAD, LeakyReLU, and other glue operators that are easy
to overlook when focusing only on the largest MAC-heavy layers.

heliaCORE responds to those Ambiq silicon needs by adding 200+ accelerated
operators and extra variants around the CMSIS-NN-compatible surface. The point is
not only to add more API names. It is to map the common model work onto the
acceleration features that are actually present in Ambiq firmware targets.

## What the acceleration paths are

| Path | What it is | Typical role |
|---|---|---|
| Scalar C | Portable C implementation for Cortex-M targets without specialized math extensions. | Baseline path for compatibility and for smaller cores such as Cortex-M0-class devices. |
| DSP extension | Packed scalar fixed-point instructions exposed through CMSIS-style intrinsics and selected with `ARM_MATH_DSP`. | Efficient int8/int16/int32 quantized kernels on Cortex-M devices that have DSP but not MVE. |
| MVE / Helium | Arm M-Profile Vector Extension, selected through MVE feature macros such as `ARM_MATH_MVEI`. | Primary high-throughput path on Cortex-M55-class devices, especially for vector-friendly AI and DSP kernels. |
| FPU / FP16 support | Floating-point hardware and compiler support used by floating-point kernels when available. | Enables fp16/fp32 operator variants; it complements the backend path rather than replacing DSP or MVE. |

<div class="workflow workflow--accel" markdown>

<div class="workflow-step" markdown>
<span class="workflow-icon">MVE</span>
<strong>128-bit vector AI/DSP acceleration</strong>
<p>Arm MVE, also known as Helium, gives Cortex-M55 a 128-bit vector engine. A
single vector can cover 16 int8 lanes, 8 int16 or fp16 lanes, or 4 int32 or fp32
lanes, with vector MAC, dot-product, saturation, predication, and lane-wise data
movement support.</p>
</div>

<div class="workflow-step" markdown>
<span class="workflow-icon">DSP</span>
<strong>Packed scalar DSP instructions</strong>
<p>The Cortex-M DSP extension accelerates fixed-point math with packed 8-bit and
16-bit operations, dual 16-bit MAC instructions, saturating arithmetic, and fast
accumulate patterns. It is still the right target for Apollo-class devices that
do not include MVE.</p>
</div>

<div class="workflow-step" markdown>
<span class="workflow-icon">Flow</span>
<strong>Whole-graph speed, not isolated tricks</strong>
<p>MVE is more than a simple intrinsic spelling. Good kernels use the hardware to
keep many lanes busy across convolution, fully connected, activation, pooling,
quantization, and layout operators so the model graph spends less time between
layers.</p>
</div>

</div>

| Capability | Cortex-M DSP paths | Cortex-M55 MVE paths |
|---|---|---|
| Execution model | Packed operations in the scalar core. | 128-bit vector execution with predication and lane-wise operations. |
| Integer work | Fixed-point int8/int16/int32 math, saturation, packing, and dual 16-bit MAC patterns. | 16 int8 lanes, 8 int16 lanes, or 4 int32 lanes per vector, with vector MAC and dot-product style kernels. |
| Floating point work | Usually not the main acceleration path for neural-network kernels. | 8 fp16 lanes or 4 fp32 lanes per vector where the target and toolchain enable those paths. |
| Best fit | Apollo-class targets without MVE and kernels where packed fixed-point DSP still moves latency. | Cortex-M55-class Apollo targets, especially int8 and int16 MAC-heavy layers plus vector-friendly graph operators. |
| Practical expectation | A meaningful improvement over plain scalar C for quantized kernels. | Kernel- and shape-dependent, but MAC-heavy int8 paths can make an up-to-8x speedup over DSP-style implementations plausible when data layout and memory traffic cooperate. |

The important firmware takeaway is that MVE changes the unit of work. Instead of
asking the core to retire one scalar multiply-accumulate at a time, optimized MVE
kernels feed a vector engine that can perform many narrow operations together and
accumulate efficiently. That is why heliaCORE treats MVE coverage as a primary
design axis for Cortex-M55-class Ambiq devices, while still preserving DSP paths
for the Apollo devices that rely on them.

## Coverage pressure from real model graphs

The counts below describe Ambiq's internal coverage target for HELIA workflows on
Ambiq devices. They are included to explain why the library invests in both
MAC-heavy kernels and the smaller operators that connect them.

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

## How to read the counts

These numbers describe coverage pressure, not benchmark performance:

| Count | Meaning |
|---|---|
| Operator types | Distinct operator families present in the model suite. |
| Unique operators | Distinct operator configurations after dtype, shape pattern, or variant differences are considered. |
| Operator instances | Total appearances across the model graphs in the suite. |

The Ambiq field-like suite is an internal workload set used to guide HELIA
coverage work on Ambiq devices. MLPerf Tiny is included as a familiar public
baseline for scale, not as a claim about relative model quality or ecosystem
coverage.

:::{note} Scope
This coverage view is based on Ambiq's internal model suite. It is
not a performance benchmark and is not a statement about the broader Arm
CMSIS-NN ecosystem. For vendor-neutral Cortex-M kernel work, Arm CMSIS-NN
remains the upstream ecosystem reference.
:::
