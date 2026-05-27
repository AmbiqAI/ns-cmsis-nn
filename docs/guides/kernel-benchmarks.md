# Kernel Benchmarks

<div class="bench-hero" markdown>

<p class="section-eyebrow">Apollo510 EVB · Cortex-M55 · 96 MHz LP mode</p>
<p class="bench-hero-title">Peak speedups at a glance</p>
<p class="bench-hero-copy">heliaCORE’s kernel mix delivers large gains on real hardware. These summary figures reflect the peak average-cycle speedups across the benchmarked kernels below.</p>

<div class="bench-stat-strip" markdown>

<div class="bench-stat-card" markdown>
<strong class="bench-stat-value">11.98×</strong>
<span class="bench-stat-label">MVE vs REF</span>
<span class="bench-stat-note">Up to 11.98× faster than the REF path</span>
</div>

<div class="bench-stat-card" markdown>
<strong class="bench-stat-value">7.83×</strong>
<span class="bench-stat-label">MVE vs DSP</span>
<span class="bench-stat-note">Up to 7.83× faster than DSP</span>
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
| `arm_convolve_s8 (contains full im2col)` | 32x32x64 k3 oc64 | 90,776,688 | 59,320,711 | 7,577,201 | 1.53x | 11.98x |
| `arm_convolve_s4 (contains full im2col)` | 32x32x64 k3 oc64 | 99,947,570 | 130,619,816 | 18,468,361 | 0.77x | 5.41x |
| `arm_convolve_s16 (contains full im2col)` | 32x32x64 k3 oc64 | 96,570,332 | 82,166,983 | 31,488,224 | 1.18x | 3.07x |
| `arm_convolve_1x1_s8_fast (contains simplified im2col)` | 32x32x64 oc64 | 11,885,922 | 8,966,843 | 1,707,351 | 1.33x | 6.96x |
| `arm_depthwise_conv_s8_opt (contains input packing)` | 32x32x64 k3 | 17,841,554 | 7,509,340 | 1,741,530 | 2.38x | 10.24x |
| `arm_depthwise_conv_s4_opt (contains input packing)` | 32x32x64 k3 | 7,984,346 | 9,767,607 | 2,188,508 | 0.82x | 3.65x |
| `arm_depthwise_conv_fast_s16 (contains input packing)` | 32x32x64 k3 | 18,069,189 | 7,571,340 | 3,861,216 | 2.39x | 4.68x |
| `arm_nn_mat_mult_nt_t_s8` | 64x512 × 256x512 | 19,959,308 | 13,725,751 | 2,086,979 | 1.45x | 9.56x |
| `arm_nn_mat_mult_nt_t_s4` | 64x512 × 256x512 | 26,937,786 | 26,723,602 | 4,197,172 | 1.01x | 6.42x |
| `arm_nn_vec_mat_mult_t_s8` | 512 × 256 | 491,641 | 331,411 | 69,076 | 1.48x | 7.12x |
| `arm_nn_vec_mat_mult_t_s16` | 512 × 256 | 562,008 | 421,257 | 99,807 | 1.33x | 5.63x |
| `arm_nn_vec_mat_mult_t_s4` | 512 × 256 | 504,554 | 588,599 | 88,622 | 0.86x | 5.69x |
| `arm_avgpool_s8` | 32x32x64 k3 | 13,228,812 | 4,057,857 | 2,135,472 | 3.26x | 6.19x |
| `arm_avgpool_s16` | 32x32x64 k3 | 6,517,842 | 4,187,804 | 2,408,673 | 1.56x | 2.71x |

## Elementwise & scalar kernels

Elementwise kernels are pure data-parallel and benefit from MVE's ability to
process 16 int8 or 8 int16 values per vector instruction.

<div class="chart-wrap" style="position:relative; max-width:760px; margin:2em auto;">
<canvas id="chart-elem-cycles" height="400"></canvas>
</div>

The x-axis reports speedup relative to REF on a logarithmic scale; values greater than 1.0x indicate lower average cycle count than REF.

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_elementwise_add_s8` | n4096 | 250,005 | 285,797 | 61,575 | 0.87x | 4.06x |
| `arm_elementwise_add_s16` | n4096 | 252,004 | 252,078 | 53,370 | 1.00x | 4.72x |
| `arm_elementwise_mul_s8` | n4096 | 98,358 | 105,093 | 22,647 | 0.94x | 4.34x |
| `arm_elementwise_mul_s16` | n4096 | 77,914 | 79,962 | 28,755 | 0.97x | 2.71x |
| `arm_elementwise_sub_s8` | n4096 | 254,030 | 285,757 | 61,632 | 0.89x | 4.12x |
| `arm_elementwise_mul_acc_s16` | n4096 | 86,175 | 90,198 | 31,851 | 0.96x | 2.71x |
| `arm_elementwise_mul_s16_s8` | n4096 | 86,098 | 84,088 | 28,778 | 1.02x | 2.99x |
| `arm_elementwise_mul_s16_batch_offset` | n4096 | 82,032 | 82,025 | 28,780 | 1.00x | 2.85x |
| `arm_add_scalar_s8` | n4096 | 131,165 | 179,461 | 42,129 | 0.73x | 3.11x |
| `arm_sub_scalar_s8` | n4096 | 172,164 | 197,856 | 42,161 | 0.87x | 4.08x |
| `arm_mul_scalar_s8` | n4096 | 90,169 | 94,308 | 20,087 | 0.96x | 4.49x |
| `arm_mul_scalar_s16` | n4096 | 77,911 | 77,908 | 27,728 | 1.00x | 2.81x |
| `arm_comparison_s8` | n4096 | 266,441 | 208,747 | 78,071 | 1.28x | 3.41x |
| `arm_comparison_s16` | n4096 | 270,764 | 237,720 | 77,986 | 1.14x | 3.47x |

## Test conditions

| Parameter | Value |
|-----------|-------|
| **Board** | Apollo510 EVB (Cortex-M55) |
| **Clock** | 96 MHz (LP mode) |
| **Timer** | DWT->CYCCNT |
| **ISA paths** | Scalar C (REF), DSP SIMD, MVE / Helium |
| **Toolchain** | arm-none-eabi-gcc 14.3.0 |
| **Optimization** | -O3, CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY |
| **Iterations** | 100 per kernel |
| **Metric** | Average CPU cycles |
