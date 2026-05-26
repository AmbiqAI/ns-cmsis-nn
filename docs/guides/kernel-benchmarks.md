# Kernel Benchmarks

<div class="bench-hero" markdown>

<p class="section-eyebrow">Apollo510 EVB · Cortex-M55 · 250 MHz HP mode</p>
<p class="bench-hero-title">Peak speedups at a glance</p>
<p class="bench-hero-copy">heliaCORE’s kernel mix delivers large gains on real hardware. These summary figures reflect the peak average-cycle speedups across the benchmarked kernels below.</p>

<div class="bench-stat-strip" markdown>

<div class="bench-stat-card" markdown>
<strong class="bench-stat-value">11.80×</strong>
<span class="bench-stat-label">MVE vs REF</span>
<span class="bench-stat-note">Up to 11.80× faster than the REF path</span>
</div>

<div class="bench-stat-card" markdown>
<strong class="bench-stat-value">7.72×</strong>
<span class="bench-stat-label">MVE vs DSP</span>
<span class="bench-stat-note">Up to 7.72× faster than DSP</span>
</div>

<div class="bench-stat-card" markdown>
<strong class="bench-stat-value">3.26×</strong>
<span class="bench-stat-label">DSP vs REF</span>
<span class="bench-stat-note">Up to 3.26× faster than the REF path</span>
</div>

</div>

</div>

This page highlights how heliaCORE performs on Apollo510 (Cortex-M55) across
the scalar reference, DSP, and MVE execution paths. Because all three paths
run on the **same SoC** at the **same clock speed**, the results provide a
clean apples-to-apples view of the performance gains available from each ISA
optimization path.

:::{note}
The REF path represents the portable C implementation of each kernel compiled
for Cortex-M55. Compiler optimizations may still generate DSP or other
architecture-specific instructions where profitable. REF therefore reflects
the practical baseline performance achievable from portable source code on
M55, rather than a strictly scalar instruction stream.
:::
 

## Convolution & matrix-multiply kernels

The largest MVE gains appear in MAC-dominated kernels where Helium's 128-bit vector engine amortizes unpacking, memory access, and accumulation overhead across wide vector operations. DSP kernels generally improve throughput on s8/s16 workloads, though some s4 kernels remain dominated by packing and unpacking overhead.

<div class="chart-wrap" style="position:relative; max-width:860px; margin:2em auto;">
<canvas id="chart-conv-cycles" height="440"></canvas>
</div>

The x-axis reports speedup relative to REF on a logarithmic scale; values greater than 1.0x indicate lower average cycle count than REF.

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_convolve_s8 (contains full im2col)` | 32x32x64 k3 oc64 | 90,903,318 | 59,462,576 | 7,704,705 | 1.53x | 11.80x |
| `arm_convolve_s4 (contains full im2col)` | 32x32x64 k3 oc64 | 100,007,039 | 130,666,354 | 18,507,250 | 0.77x | 5.40x |
| `arm_convolve_s16 (contains full im2col)` | 32x32x64 k3 oc64 | 96,838,945 | 82,449,008 | 31,782,453 | 1.17x | 3.05x |
| `arm_convolve_1x1_s8_fast (contains simplified im2col)` | 32x32x64 oc64 | 12,001,651 | 9,097,394 | 1,724,089 | 1.32x | 6.96x |
| `arm_depthwise_conv_s8_opt (contains input packing)` | 32x32x64 k3 | 17,846,823 | 7,513,771 | 1,751,452 | 2.38x | 10.19x |
| `arm_depthwise_conv_s4_opt (contains input packing)` | 32x32x64 k3 | 7,993,787 | 9,758,291 | 2,197,534 | 0.82x | 3.64x |
| `arm_depthwise_conv_fast_s16 (contains input packing)` | 32x32x64 k3 | 18,346,309 | 7,591,946 | 3,897,696 | 2.42x | 4.71x |
| `arm_nn_mat_mult_nt_t_s8` | 64x512 × 256x512 | 20,046,490 | 13,818,437 | 2,419,506 | 1.45x | 8.29x |
| `arm_nn_mat_mult_nt_t_s4` | 64x512 × 256x512 | 26,983,473 | 26,771,230 | 4,306,744 | 1.01x | 6.27x |
| `arm_nn_vec_mat_mult_t_s8` | 512 × 256 | 527,300 | 368,831 | 105,789 | 1.43x | 4.98x |
| `arm_nn_vec_mat_mult_t_s16` | 512 × 256 | 578,394 | 459,403 | 137,985 | 1.26x | 4.19x |
| `arm_nn_vec_mat_mult_t_s4` | 512 × 256 | 506,512 | 590,310 | 90,381 | 0.86x | 5.60x |
| `arm_avgpool_s8` | 32x32x64 k3 | 13,247,847 | 4,058,600 | 2,153,880 | 3.26x | 6.15x |
| `arm_avgpool_s16` | 32x32x64 k3 | 6,553,317 | 4,197,299 | 2,445,830 | 1.56x | 2.68x |

## Elementwise & scalar kernels

Elementwise kernels are pure data-parallel and benefit from MVE's ability to
process 16 int8 or 8 int16 values per vector instruction.

<div class="chart-wrap" style="position:relative; max-width:760px; margin:2em auto;">
<canvas id="chart-elem-cycles" height="400"></canvas>
</div>

The x-axis reports speedup relative to REF on a logarithmic scale; values greater than 1.0x indicate lower average cycle count than REF.

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_elementwise_add_s8` | n4096 | 249,931 | 285,797 | 61,572 | 0.87x | 4.06x |
| `arm_elementwise_add_s16` | n4096 | 251,998 | 252,173 | 53,365 | 1.00x | 4.72x |
| `arm_elementwise_mul_s8` | n4096 | 98,353 | 105,024 | 22,643 | 0.94x | 4.34x |
| `arm_elementwise_mul_s16` | n4096 | 77,907 | 79,955 | 28,746 | 0.97x | 2.71x |
| `arm_elementwise_sub_s8` | n4096 | 254,198 | 285,758 | 61,566 | 0.89x | 4.13x |
| `arm_elementwise_mul_acc_s16` | n4096 | 86,096 | 90,193 | 31,832 | 0.95x | 2.70x |
| `arm_elementwise_mul_s16_s8` | n4096 | 86,092 | 84,082 | 28,773 | 1.02x | 2.99x |
| `arm_elementwise_mul_s16_batch_offset` | n4096 | 82,025 | 82,018 | 28,773 | 1.00x | 2.85x |
| `arm_add_scalar_s8` | n4096 | 131,157 | 179,383 | 42,124 | 0.73x | 3.11x |
| `arm_sub_scalar_s8` | n4096 | 172,157 | 198,013 | 42,157 | 0.87x | 4.08x |
| `arm_mul_scalar_s8` | n4096 | 90,166 | 94,322 | 20,083 | 0.96x | 4.49x |
| `arm_mul_scalar_s16` | n4096 | 77,899 | 77,895 | 27,718 | 1.00x | 2.81x |
| `arm_comparison_s8` | n4096 | 266,729 | 208,667 | 77,994 | 1.28x | 3.42x |
| `arm_comparison_s16` | n4096 | 266,891 | 237,716 | 77,983 | 1.12x | 3.42x |

## Test conditions

| Parameter | Value |
|-----------|-------|
| **Board** | Apollo510 EVB (Cortex-M55) |
| **Clock** | 250 MHz (HP mode) |
| **Timer** | DWT->CYCCNT |
| **ISA paths** | Scalar C (REF), DSP SIMD, MVE / Helium |
| **Toolchain** | arm-none-eabi-gcc 14.3.0 |
| **Optimization** | -O3, CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY |
| **Iterations** | 100 per kernel |
| **Metric** | Average CPU cycles |
