// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
/**
 * Apollo510 kernel benchmark results, as published in the results tables on
 * guides/kernel-benchmarks.
 *
 * Every row of those tables is carried here whole, including the columns the
 * charts do not plot, so the charts are fed from a record of the measurement
 * rather than from a subset of it. The Sphinx charts this replaces scraped the
 * numbers back out of the rendered tables, so the two can still be compared
 * value for value.
 *
 * Conditions: Apollo510 EVB (Cortex-M55), 96 MHz LP mode, DWT->CYCCNT,
 * arm-none-eabi-gcc 14.3.0 at -O3, 100 iterations per kernel.
 */

/** Average CPU cycles per call on each ISA path. */
export interface KernelCycles {
  readonly ref: number;
  readonly dsp: number;
  readonly mve: number;
}

export interface KernelBenchmark {
  /** Kernel symbol, with the qualifying note the results table carries. */
  readonly kernel: string;
  /** Short form of the symbol, used on the chart's category axis. */
  readonly label: string;
  /** Workload the cycle counts were measured on. */
  readonly shape: string;
  readonly cycles: KernelCycles;
}

/*
 * The ISA paths. The order is the one the names sort in, not the one the
 * results tables read in: the chart part takes the order of the bars in a
 * group from the sorted series names and the order of the legend from the
 * order of the rows, so rows written any other way label the chart wrongly.
 */
export const ISA_PATHS = {
  dsp: 'DSP (GCC)',
  mve: 'MVE (GCC)',
  ref: 'Reference (GCC)',
} as const;

export type IsaPath = keyof typeof ISA_PATHS;

/** One bar: a kernel, an ISA path, and that path's speedup against REF. */
export interface KernelSpeedup {
  readonly kernel: string;
  readonly path: string;
  readonly speedup: number;
}

/**
 * Speedup against the REF path, REF cycles over the path's own. REF is 1.0 by
 * construction and is plotted so the baseline the other two are read against
 * is on the chart rather than in the reader's head.
 */
export function refSpeedups(
  rows: readonly KernelBenchmark[],
): KernelSpeedup[] {
  const paths = Object.keys(ISA_PATHS) as IsaPath[];
  return rows.flatMap((row) =>
    paths.map((path) => ({
      kernel: row.label,
      path: ISA_PATHS[path],
      speedup: row.cycles.ref / row.cycles[path],
    })),
  );
}

export const CONVOLUTION_KERNELS: readonly KernelBenchmark[] = [
  { kernel: 'arm_convolve_s8 (contains full im2col)', label: 'convolve_s8', shape: '32x32x64 k3 oc64', cycles: { ref: 90776688, dsp: 59320711, mve: 7577201 } },
  { kernel: 'arm_convolve_s4 (contains full im2col)', label: 'convolve_s4', shape: '32x32x64 k3 oc64', cycles: { ref: 99947570, dsp: 130619816, mve: 18468361 } },
  { kernel: 'arm_convolve_s16 (contains full im2col)', label: 'convolve_s16', shape: '32x32x64 k3 oc64', cycles: { ref: 96570332, dsp: 82166983, mve: 31488224 } },
  { kernel: 'arm_convolve_1x1_s8_fast (contains simplified im2col)', label: 'conv_1x1_s8', shape: '32x32x64 oc64', cycles: { ref: 11885922, dsp: 8966843, mve: 1707351 } },
  { kernel: 'arm_depthwise_conv_s8_opt (contains input packing)', label: 'dw_conv_s8', shape: '32x32x64 k3', cycles: { ref: 17841554, dsp: 7509340, mve: 1741530 } },
  { kernel: 'arm_depthwise_conv_s4_opt (contains input packing)', label: 'dw_conv_s4', shape: '32x32x64 k3', cycles: { ref: 7984346, dsp: 9767607, mve: 2188508 } },
  { kernel: 'arm_depthwise_conv_fast_s16 (contains input packing)', label: 'dw_conv_s16', shape: '32x32x64 k3', cycles: { ref: 18069189, dsp: 7571340, mve: 3861216 } },
  { kernel: 'arm_nn_mat_mult_nt_t_s8', label: 'mat_mult_s8', shape: '64x512 × 256x512', cycles: { ref: 19959308, dsp: 13725751, mve: 2086979 } },
  { kernel: 'arm_nn_mat_mult_nt_t_s4', label: 'mat_mult_s4', shape: '64x512 × 256x512', cycles: { ref: 26937786, dsp: 26723602, mve: 4197172 } },
  { kernel: 'arm_nn_vec_mat_mult_t_s8', label: 'vec_mat_s8', shape: '512 × 256', cycles: { ref: 491641, dsp: 331411, mve: 69076 } },
  { kernel: 'arm_nn_vec_mat_mult_t_s16', label: 'vec_mat_s16', shape: '512 × 256', cycles: { ref: 562008, dsp: 421257, mve: 99807 } },
  { kernel: 'arm_nn_vec_mat_mult_t_s4', label: 'vec_mat_s4', shape: '512 × 256', cycles: { ref: 504554, dsp: 588599, mve: 88622 } },
  { kernel: 'arm_avgpool_s8', label: 'avgpool_s8', shape: '32x32x64 k3', cycles: { ref: 13228812, dsp: 4057857, mve: 2135472 } },
  { kernel: 'arm_avgpool_s16', label: 'avgpool_s16', shape: '32x32x64 k3', cycles: { ref: 6517842, dsp: 4187804, mve: 2408673 } },
];

export const ELEMENTWISE_KERNELS: readonly KernelBenchmark[] = [
  { kernel: 'arm_elementwise_add_s8', label: 'add_s8', shape: 'n4096', cycles: { ref: 250005, dsp: 285797, mve: 61575 } },
  { kernel: 'arm_elementwise_add_s16', label: 'add_s16', shape: 'n4096', cycles: { ref: 252004, dsp: 252078, mve: 53370 } },
  { kernel: 'arm_elementwise_mul_s8', label: 'mul_s8', shape: 'n4096', cycles: { ref: 98358, dsp: 105093, mve: 22647 } },
  { kernel: 'arm_elementwise_mul_s16', label: 'mul_s16', shape: 'n4096', cycles: { ref: 77914, dsp: 79962, mve: 28755 } },
  { kernel: 'arm_elementwise_sub_s8', label: 'sub_s8', shape: 'n4096', cycles: { ref: 254030, dsp: 285757, mve: 61632 } },
  { kernel: 'arm_elementwise_mul_acc_s16', label: 'mul_acc_s16', shape: 'n4096', cycles: { ref: 86175, dsp: 90198, mve: 31851 } },
  { kernel: 'arm_elementwise_mul_s16_s8', label: 'mul_s16_s8', shape: 'n4096', cycles: { ref: 86098, dsp: 84088, mve: 28778 } },
  { kernel: 'arm_elementwise_mul_s16_batch_offset', label: 'mul_s16_batch', shape: 'n4096', cycles: { ref: 82032, dsp: 82025, mve: 28780 } },
  { kernel: 'arm_add_scalar_s8', label: 'add_scalar_s8', shape: 'n4096', cycles: { ref: 131165, dsp: 179461, mve: 42129 } },
  { kernel: 'arm_sub_scalar_s8', label: 'sub_scalar_s8', shape: 'n4096', cycles: { ref: 172164, dsp: 197856, mve: 42161 } },
  { kernel: 'arm_mul_scalar_s8', label: 'mul_scalar_s8', shape: 'n4096', cycles: { ref: 90169, dsp: 94308, mve: 20087 } },
  { kernel: 'arm_mul_scalar_s16', label: 'mul_scalar_s16', shape: 'n4096', cycles: { ref: 77911, dsp: 77908, mve: 27728 } },
  { kernel: 'arm_comparison_s8', label: 'comparison_s8', shape: 'n4096', cycles: { ref: 266441, dsp: 208747, mve: 78071 } },
  { kernel: 'arm_comparison_s16', label: 'comparison_s16', shape: 'n4096', cycles: { ref: 270764, dsp: 237720, mve: 77986 } },
];
