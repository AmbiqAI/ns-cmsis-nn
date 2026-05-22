# Kernel Benchmarks

Cycle-count benchmarks for heliaCORE kernels on Apollo510 (Cortex-M55),
comparing the scalar reference, DSP, and MVE acceleration paths. All three
paths run on the **same SoC** at the **same clock speed**, so the comparison
is a direct apples-to-apples ISA measurement.

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

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_convolve_s8 (contains full im2col)` | 32x32x64 k3 oc64 | 91,508,570 | 59,724,899 | 7,855,297 | 1.53x | 11.65x |
| `arm_convolve_s4 (contains full im2col)` | 32x32x64 k3 oc64 | 100,429,550 | 129,526,146 | 18,506,232 | 0.78x | 5.43x |
| `arm_convolve_s16 (contains full im2col)` | 32x32x64 k3 oc64 | 95,924,158 | 82,518,621 | 31,815,146 | 1.16x | 3.02x |
| `arm_convolve_1x1_s8_fast (contains simplified im2col)` | 32x32x64 oc64 | 12,455,209 | 9,526,873 | 1,839,405 | 1.31x | 6.77x |
| `arm_depthwise_conv_s8_opt (contains input packing)` | 32x32x64 k3 | 18,716,382 | 7,794,105 | 1,751,320 | 2.40x | 10.69x |
| `arm_depthwise_conv_s4_opt (contains input packing)` | 32x32x64 k3 | 8,341,819 | 9,922,017 | 2,198,403 | 0.84x | 3.79x |
| `arm_depthwise_conv_fast_s16 (contains input packing)` | 32x32x64 k3 | 18,160,003 | 7,619,179 | 3,903,131 | 2.38x | 4.65x |
| `arm_nn_mat_mult_nt_t_s8` | 64x512 × 256x512 | 20,159,721 | 13,924,038 | 2,452,107 | 1.45x | 8.22x |
| `arm_nn_mat_mult_nt_t_s4` | 64x512 × 256x512 | 27,086,756 | 26,504,290 | 4,305,830 | 1.02x | 6.29x |
| `arm_nn_vec_mat_mult_t_s8` | 512 × 256 | 528,987 | 369,311 | 105,856 | 1.43x | 5.00x |
| `arm_nn_vec_mat_mult_t_s16` | 512 × 256 | 578,682 | 459,348 | 137,337 | 1.26x | 4.21x |
| `arm_nn_vec_mat_mult_t_s4` | 512 × 256 | 507,839 | 591,498 | 90,839 | 0.86x | 5.59x |
| `arm_avgpool_s8` | 32x32x64 k3 | 13,245,517 | 4,060,374 | 2,153,231 | 3.26x | 6.15x |
| `arm_avgpool_s16` | 32x32x64 k3 | 6,552,871 | 4,200,737 | 2,445,854 | 1.56x | 2.68x |

## Elementwise & scalar kernels

Elementwise kernels are pure data-parallel and benefit from MVE's ability to
process 16 int8 or 8 int16 values per vector instruction.

<div class="chart-wrap" style="position:relative; max-width:760px; margin:2em auto;">
<canvas id="chart-elem-cycles" height="400"></canvas>
</div>

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_elementwise_add_s8` | n4096 | 385,147 | 302,208 | 61,572 | 1.27x | 6.26x |
| `arm_elementwise_add_s16` | n4096 | 331,969 | 331,972 | 53,365 | 1.00x | 6.22x |
| `arm_elementwise_mul_s8` | n4096 | 143,599 | 165,137 | 22,645 | 0.87x | 6.34x |
| `arm_elementwise_mul_s16` | n4096 | 110,680 | 151,638 | 28,750 | 0.73x | 3.85x |
| `arm_elementwise_sub_s8` | n4096 | 381,344 | 314,493 | 61,567 | 1.21x | 6.19x |
| `arm_elementwise_mul_acc_s16` | n4096 | 133,212 | 155,742 | 31,998 | 0.86x | 4.16x |
| `arm_elementwise_mul_s16_s8` | n4096 | 131,162 | 141,768 | 28,790 | 0.93x | 4.56x |
| `arm_elementwise_mul_s16_batch_offset` | n4096 | 121,098 | 133,235 | 28,773 | 0.91x | 4.21x |
| `arm_add_scalar_s8` | n4096 | 270,440 | 218,657 | 42,120 | 1.24x | 6.42x |
| `arm_sub_scalar_s8` | n4096 | 291,139 | 218,873 | 42,169 | 1.33x | 6.90x |
| `arm_mul_scalar_s8` | n4096 | 135,228 | 156,778 | 20,081 | 0.86x | 6.73x |
| `arm_mul_scalar_s16` | n4096 | 114,765 | 137,300 | 27,718 | 0.84x | 4.14x |
| `arm_comparison_s8` | n4096 | 327,815 | 244,901 | 77,994 | 1.34x | 4.20x |
| `arm_comparison_s16` | n4096 | 327,814 | 274,791 | 77,981 | 1.19x | 4.20x |

## ATfE vs GCC on MVE

The ATfE (LLVM/Clang) toolchain often generates better MVE code than GCC.
This chart compares GCC and ATfE cycle counts for the MVE path on the same
Apollo510 hardware.

<div class="chart-wrap" style="position:relative; max-width:820px; margin:2em auto;">
<canvas id="chart-atfe-cycles" height="440"></canvas>
</div>

| Kernel | MVE (GCC) | MVE (ATfE) | ATfE speedup |
|--------|----------:|-----------:|-------------:|
| `arm_convolve_s8` | 7,855,297 | 7,500,319 | 1.05x |
| `arm_convolve_s4` | 18,506,232 | 14,976,687 | 1.24x |
| `arm_convolve_s16` | 31,815,146 | 16,901,027 | 1.88x |
| `arm_convolve_1x1_s8_fast` | 1,839,405 | 1,534,130 | 1.20x |
| `arm_depthwise_conv_s8_opt` | 1,751,320 | 1,584,841 | 1.11x |
| `arm_depthwise_conv_s4_opt` | 2,198,403 | 2,163,861 | 1.02x |
| `arm_depthwise_conv_fast_s16` | 3,903,131 | 4,009,775 | 0.97x |
| `arm_nn_mat_mult_nt_t_s8` | 2,452,107 | 2,357,922 | 1.04x |
| `arm_nn_mat_mult_nt_t_s4` | 4,305,830 | 3,531,101 | 1.22x |
| `arm_nn_vec_mat_mult_t_s8` | 105,856 | 108,992 | 0.97x |
| `arm_nn_vec_mat_mult_t_s16` | 137,337 | 135,408 | 1.01x |
| `arm_nn_vec_mat_mult_t_s4` | 90,839 | 62,916 | 1.44x |
| `arm_avgpool_s8` | 2,153,231 | 1,724,589 | 1.25x |
| `arm_avgpool_s16` | 2,445,854 | 2,043,942 | 1.20x |


## Test conditions

| Parameter | Value |
|-----------|-------|
| **Board** | Apollo510 EVB (Cortex-M55) |
| **Clock** | 250 MHz (HP mode) |
| **Timer** | DWT->CYCCNT |
| **ISA paths** | Scalar C (REF), DSP SIMD, MVE / Helium |
| **Toolchains** | arm-none-eabi-gcc 14.3.0, ATfE 22.1.0 |
| **Optimization** | -O3 |
| **Iterations** | 100 per kernel |
| **Metric** | Average CPU cycles |
