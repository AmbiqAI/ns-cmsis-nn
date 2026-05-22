# Kernel Benchmarks

Cycle-count benchmarks for heliaCORE kernels on Apollo510 (Cortex-M55),
comparing the scalar reference, DSP, and MVE acceleration paths. All three
paths run on the **same SoC** at the **same clock speed**, so the comparison
is a direct apples-to-apples ISA measurement.

:::{note} What these numbers measure
All values are **average CPU cycles** collected over 100 iterations on an
Apollo510 EVB running at **250 MHz** (HP mode). Because every path runs on
the same hardware, the only variable is the ISA code path. To convert to
microseconds: `time_µs = cycles / 250`.
:::

## Convolution & matrix-multiply kernels

The biggest MVE gains appear in MAC-dominated kernels where Helium's 128-bit
vector unit processes 16 × int8 elements per cycle versus DSP's dual-MAC
and scalar C's single-element loops.

<div class="chart-wrap" style="position:relative; max-width:860px; margin:2em auto;">
<canvas id="chart-conv-cycles" height="440"></canvas>
</div>

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_convolve_s8` | 32x32x64 k3 oc64 | 12,189 | 8,096 | 1,106 | 1.51x | 11.02x |
| `arm_convolve_s4` | 32x32x64 k3 oc64 | 13,206 | 17,028 | 2,432 | 0.78x | 5.43x |
| `arm_convolve_s16` | 32x32x64 k3 oc64 | 12,964 | 11,283 | 4,177 | 1.15x | 3.10x |
| `arm_convolve_1x1_s8_fast` | 32x32x64 oc64 | 1,635 | 1,273 | 286 | 1.28x | 5.72x |
| `arm_depthwise_conv_s8_opt` | 32x32x64 k3 | 2,453 | 1,070 | 236 | 2.29x | 10.39x |
| `arm_depthwise_conv_s4_opt` | 32x32x64 k3 | 1,134 | 1,345 | 294 | 0.84x | 3.86x |
| `arm_depthwise_conv_fast_s16` | 32x32x64 k3 | 2,374 | 1,414 | 914 | 1.68x | 2.60x |
| `arm_nn_mat_mult_nt_t_s8` | 64x512 × 256x512 | 2,643 | 1,860 | 396 | 1.42x | 6.67x |
| `arm_nn_mat_mult_nt_t_s4` | 64x512 × 256x512 | 3,551 | 3,493 | 634 | 1.02x | 5.60x |
| `arm_nn_vec_mat_mult_t_s8` | 512 × 256 | 69 | 49 | 15 | 1.41x | 4.60x |
| `arm_nn_vec_mat_mult_t_s16` | 512 × 256 | 76 | 60 | 18 | 1.27x | 4.22x |
| `arm_nn_vec_mat_mult_t_s4` | 512 × 256 | 66 | 77 | 13 | 0.86x | 5.08x |
| `arm_avgpool_s8` | 32x32x64 k3 | 1,736 | 535 | 288 | 3.24x | 6.03x |
| `arm_avgpool_s16` | 32x32x64 k3 | 858 | 551 | 318 | 1.56x | 2.70x |

## Elementwise & scalar kernels

Elementwise kernels are pure data-parallel and benefit from MVE's ability to
process 16 int8 or 8 int16 values per vector instruction.

<div class="chart-wrap" style="position:relative; max-width:760px; margin:2em auto;">
<canvas id="chart-elem-cycles" height="400"></canvas>
</div>

| Kernel | Shape | REF (GCC) | DSP (GCC) | MVE (GCC) | DSP speedup | MVE speedup |
|--------|-------|----------:|----------:|----------:|------------:|------------:|
| `arm_elementwise_add_s8` | n4096 | 50 | 39 | 8 | 1.28x | 6.25x |
| `arm_elementwise_add_s16` | n4096 | 43 | 43 | 7 | 1.00x | 6.14x |
| `arm_elementwise_mul_s8` | n4096 | 18 | 21 | 3 | 0.86x | 6.00x |
| `arm_elementwise_mul_s16` | n4096 | 14 | 20 | 3 | 0.70x | 4.67x |
| `arm_elementwise_sub_s8` | n4096 | 49 | 41 | 8 | 1.20x | 6.13x |
| `arm_elementwise_mul_acc_s16` | n4096 | 17 | 20 | 4 | 0.85x | 4.25x |
| `arm_elementwise_mul_s16_s8` | n4096 | 17 | 19 | 3 | 0.89x | 5.67x |
| `arm_elementwise_mul_s16_batch_offset` | n4096 | 16 | 18 | 3 | 0.89x | 5.33x |
| `arm_add_scalar_s8` | n4096 | 36 | 28 | 5 | 1.29x | 7.20x |
| `arm_sub_scalar_s8` | n4096 | 37 | 28 | 5 | 1.32x | 7.40x |
| `arm_mul_scalar_s8` | n4096 | 17 | 20 | 2 | 0.85x | 8.50x |
| `arm_mul_scalar_s16` | n4096 | 15 | 17 | 3 | 0.88x | 5.00x |
| `arm_comparison_s8` | n4096 | 43 | 32 | 10 | 1.34x | 4.30x |
| `arm_comparison_s16` | n4096 | 43 | 36 | 10 | 1.19x | 4.30x |

## ATfE vs GCC on MVE

The ATfE (LLVM/Clang) toolchain often generates better MVE code than GCC.
This chart compares GCC and ATfE cycle counts for the MVE path on the same
Apollo510 hardware.

<div class="chart-wrap" style="position:relative; max-width:820px; margin:2em auto;">
<canvas id="chart-atfe-cycles" height="440"></canvas>
</div>

| Kernel | MVE (GCC) | MVE (ATfE) | ATfE speedup |
|--------|----------:|-----------:|-------------:|
| `arm_convolve_s8` | 1,106 | 1,009 | 1.10x |
| `arm_convolve_s4` | 2,432 | 1,971 | 1.23x |
| `arm_convolve_s16` | 4,177 | 2,188 | 1.91x |
| `arm_convolve_1x1_s8_fast` | 286 | 222 | 1.29x |
| `arm_depthwise_conv_s8_opt` | 236 | 212 | 1.11x |
| `arm_depthwise_conv_s4_opt` | 294 | 288 | 1.02x |
| `arm_depthwise_conv_fast_s16` | 914 | 1,048 | 0.87x |
| `arm_nn_mat_mult_nt_t_s8` | 396 | 337 | 1.18x |
| `arm_nn_mat_mult_nt_t_s4` | 634 | 496 | 1.28x |
| `arm_nn_vec_mat_mult_t_s8` | 15 | 14 | 1.07x |
| `arm_nn_vec_mat_mult_t_s16` | 18 | 17 | 1.06x |
| `arm_nn_vec_mat_mult_t_s4` | 13 | 8 | 1.63x |
| `arm_avgpool_s8` | 288 | 230 | 1.25x |
| `arm_avgpool_s16` | 318 | 257 | 1.24x |


## Test conditions

| Parameter | Value |
|-----------|-------|
| **Board** | Apollo510 EVB (Cortex-M55) |
| **Clock** | 250 MHz (HP mode) |
| **ISA paths** | Scalar C (REF), DSP SIMD, MVE / Helium |
| **Toolchains** | arm-none-eabi-gcc 14.3.0, ATfE 22.1.0 |
| **Iterations** | 100 per kernel |
| **Metric** | Average CPU cycles |
