/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_1_x_n_f16.c
 * Description:  Dedicated float16 1xN convolution
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_conv_opt_common.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS (8)
        #define ARM_NN_CONV_1XN_F16_MVE_DUAL_BLOCK_ROWS (16)
        #define ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS (4)
        #define ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8)
        #define ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_DUAL ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_16)
        #define ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_SUB ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_4)
    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float16_t arm_convolve_1_x_n_dot_f16(const float16_t *lhs, const float16_t *rhs, int32_t len)
{
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
    for (int32_t i = 0; i < len; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(len - i));
        vacc = vfmaq_m(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p), p);
    }
    return arm_nn_vec_reduce_add_f16(vacc);
}

/* `n` elements of a dot product into fresh float16 lanes (element i goes to lane i % 8); advances both pointers. */
__STATIC_FORCEINLINE float16x8_t arm_convolve_1_x_n_dot_block_f16(const float16_t **lhs,
                                                                  const float16_t **rhs,
                                                                  int32_t n)
{
    const float16_t *pl = *lhs;
    const float16_t *pr = *rhs;
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
    for (int32_t i = 0; i < n; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(n - i));
        vacc = vfmaq_m(vacc, vld1q_z(pl + i, p), vld1q_z(pr + i, p), p);
    }
    *lhs = pl + n;
    *rhs = pr + n;
    return vacc;
}

/* Bias plus one dot product of more than ARM_NN_F16_ACC_BLOCK elements, blockwise (#586): each lane sums at most
 * ARM_NN_F16_ACC_BLOCK taps (8 * that many elements) in float16; each block's lanes are folded into float32 pair
 * accumulators (arm_nn_f16_fold_pairs_f32), summed once (arm_nn_f16_pairs_sum_f32); the bias is added in float32
 * and the total rounds to float16 once. */
__STATIC_FORCEINLINE _Float16
arm_convolve_1_x_n_dot_fold_f16(const float16_t *lhs, const float16_t *rhs, const float16_t *bias, int32_t len)
{
    const int32_t span = ARM_NN_F16_ACC_BLOCK * 8;
    float32x4_t acc = vdupq_n_f32(0.0f); /* set by the first fold */

    if (len <= span)
    {
        arm_nn_f16_fold_pairs_f32(&acc, arm_convolve_1_x_n_dot_block_f16(&lhs, &rhs, len), true);
    }
    else
    {
        bool first = true;
        for (int32_t rem = len; rem > 0;)
        {
            const int32_t n = (rem > span) ? span : rem;
            rem -= n;
            arm_nn_f16_fold_pairs_f32(&acc, arm_convolve_1_x_n_dot_block_f16(&lhs, &rhs, n), first);
            first = false;
        }
    }
    return (_Float16)((bias ? (float32_t)*bias : 0.0f) + arm_nn_f16_pairs_sum_f32(acc));
}

/* Gather-kernel inner loops (one output column per lane, per-k): `n` taps onto the given accumulators; advance the
 * lhs and rhs pointers past them. */
__STATIC_FORCEINLINE void arm_convolve_1_x_n_gather16_f16(const float16_t **lhs,
                                                          const float16_t **rhs,
                                                          uint16x8_t offsets,
                                                          uint16x8_t offsets_hi,
                                                          int32_t n,
                                                          float16x8_t *vacc_lo,
                                                          float16x8_t *vacc_hi)
{
    const float16_t *pl = *lhs;
    const float16_t *pr = *rhs;
    float16x8_t lo = *vacc_lo;
    float16x8_t hi = *vacc_hi;
    for (int32_t k = 0; k < n; ++k)
    {
        const float16_t lhs_v = pl[k];
        const float16x8_t vrhs_lo = vldrhq_gather_shifted_offset(pr + k, offsets);
        const float16x8_t vrhs_hi = vldrhq_gather_shifted_offset(pr + k, offsets_hi);
        lo = vfmaq(lo, vrhs_lo, lhs_v);
        hi = vfmaq(hi, vrhs_hi, lhs_v);
    }
    *lhs = pl + n;
    *rhs = pr + n;
    *vacc_lo = lo;
    *vacc_hi = hi;
}

__STATIC_FORCEINLINE float16x8_t arm_convolve_1_x_n_gather8_f16(const float16_t **lhs,
                                                                const float16_t **rhs,
                                                                uint16x8_t offsets,
                                                                int32_t n,
                                                                float16x8_t vacc)
{
    const float16_t *pl = *lhs;
    const float16_t *pr = *rhs;
    for (int32_t k = 0; k < n; ++k)
    {
        const float16x8_t vrhs = vldrhq_gather_shifted_offset(pr + k, offsets);
        vacc = vfmaq(vacc, vrhs, pl[k]);
    }
    *lhs = pl + n;
    *rhs = pr + n;
    return vacc;
}

__STATIC_FORCEINLINE float16x8_t arm_convolve_1_x_n_gather4_f16(const float16_t **lhs,
                                                                const float16_t **rhs,
                                                                uint16x8_t offsets,
                                                                mve_pred16_t p,
                                                                int32_t n,
                                                                float16x8_t vacc)
{
    const float16_t *pl = *lhs;
    const float16_t *pr = *rhs;
    for (int32_t k = 0; k < n; ++k)
    {
        const float16x8_t vrhs = vldrhq_gather_shifted_offset_z(pr + k, offsets, p);
        vacc = vfmaq(vacc, vrhs, pl[k]);
    }
    *lhs = pl + n;
    *rhs = pr + n;
    return vacc;
}
    #else
/* Scalar leg accumulates in float32; the caller adds the bias and rounds to f16 once (#449, #465). */
__STATIC_INLINE float32_t arm_convolve_1_x_n_dot_f16(const float16_t *lhs, const float16_t *rhs, int32_t len)
{
    float32_t acc = 0.0f;
    for (int32_t i = 0; i < len; ++i)
    {
        acc += (float32_t)lhs[i] * (float32_t)rhs[i];
    }
    return acc;
}
    #endif

/* `block` is ARM_NN_F16_ACC_BLOCK (blockwise, #586) or ARM_NN_F16_ACC_BLOCK_NONE (float16 lanes throughout) at every
 * call site. Gather kernels: the bias opens the first block of at most `block` taps, which is the float16-lane loop;
 * every further block of at most `block` taps starts from zero and is widened into per-lane float32 accumulators
 * that round to float16 once. */
__STATIC_FORCEINLINE arm_cmsis_nn_status
arm_convolve_1_x_n_mat_mult_nt_t_strided_body_f16(const float16_t *__RESTRICT lhs,
                                                  const float16_t *__RESTRICT rhs,
                                                  const float16_t *__RESTRICT bias,
                                                  float16_t *__RESTRICT dst,
                                                  int32_t lhs_rows,
                                                  int32_t rhs_rows,
                                                  int32_t rhs_cols,
                                                  int32_t lhs_cols_offset,
                                                  int32_t row_address_offset,
                                                  float16_t activation_min,
                                                  float16_t activation_max,
                                                  const int32_t block)
{
    (void)block;
    if (!lhs || !rhs || !dst || lhs_rows <= 0 || rhs_rows <= 0 || rhs_cols <= 0 || lhs_cols_offset <= 0 ||
        row_address_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* Taps of the first (bias-opened) block; more than that only when folding. */
    const int32_t first_n = (rhs_cols > block) ? block : rhs_cols;
    /* First column left to the remainder loop (the same for every lhs row). */
    int32_t c_tail = 0;
    /* Fewer rhs rows than the smallest gather group: blockwise, everything is remainder columns. */
    const int32_t gather_lhs_rows =
        (rhs_cols > block && rhs_rows < ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS) ? 0 : lhs_rows;
    #else
    const int32_t gather_lhs_rows = lhs_rows;
    #endif

    for (int32_t r = 0; r < gather_lhs_rows; ++r)
    {
        const float16_t *lhs_row = lhs + (size_t)r * lhs_cols_offset;
        float16_t *dst_row = dst + (size_t)r * row_address_offset;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t c = 0;
        if (rhs_cols <= ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS)
        {
            const uint16_t rhs_cols_u16 = (uint16_t)rhs_cols;
            const uint16x8_t offsets = vmulq(vidupq_u16((uint32_t)0, 1), rhs_cols_u16);
            const float16x8_t vmin = vdupq_n_f16(activation_min);
            const float16x8_t vmax = vdupq_n_f16(activation_max);

            if (rhs_cols <= ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_DUAL)
            {
                const uint16x8_t offsets_hi =
                    vaddq(offsets, (uint16_t)(ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS * rhs_cols_u16));

                for (; c + ARM_NN_CONV_1XN_F16_MVE_DUAL_BLOCK_ROWS <= rhs_rows;
                     c += ARM_NN_CONV_1XN_F16_MVE_DUAL_BLOCK_ROWS)
                {
                    const float16_t *rhs_block = rhs + (size_t)c * rhs_cols;
                    float16x8_t vacc_lo = bias ? vld1q(bias + c) : vdupq_n_f16((float16_t)0.0f);
                    float16x8_t vacc_hi =
                        bias ? vld1q(bias + c + ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS) : vdupq_n_f16((float16_t)0.0f);

                    const float16_t *pl = lhs_row;
                    const float16_t *pr = rhs_block;
                    arm_convolve_1_x_n_gather16_f16(&pl, &pr, offsets, offsets_hi, first_n, &vacc_lo, &vacc_hi);
                    if (rhs_cols > first_n)
                    {
                        float32x4_t lo_even = arm_nn_vcvtbq_f32_f16(vacc_lo);
                        float32x4_t lo_odd = arm_nn_vcvttq_f32_f16(vacc_lo);
                        float32x4_t hi_even = arm_nn_vcvtbq_f32_f16(vacc_hi);
                        float32x4_t hi_odd = arm_nn_vcvttq_f32_f16(vacc_hi);
                        for (int32_t rem = rhs_cols - first_n; rem > 0;)
                        {
                            const int32_t n = (rem > block) ? block : rem;
                            rem -= n;
                            vacc_lo = vdupq_n_f16((float16_t)0.0f);
                            vacc_hi = vdupq_n_f16((float16_t)0.0f);
                            arm_convolve_1_x_n_gather16_f16(&pl, &pr, offsets, offsets_hi, n, &vacc_lo, &vacc_hi);
                            arm_nn_f16_fold_lanes_f32(&lo_even, &lo_odd, vacc_lo, false);
                            arm_nn_f16_fold_lanes_f32(&hi_even, &hi_odd, vacc_hi, false);
                        }
                        vacc_lo = arm_nn_f16_narrow_lanes_f32(lo_even, lo_odd);
                        vacc_hi = arm_nn_f16_narrow_lanes_f32(hi_even, hi_odd);
                    }

                    vacc_lo = arm_nn_clamp_mve_f16(vacc_lo, vmin, vmax);
                    vacc_hi = arm_nn_clamp_mve_f16(vacc_hi, vmin, vmax);
                    vst1q(dst_row + c, vacc_lo);
                    vst1q(dst_row + c + ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS, vacc_hi);
                }
            }

            for (; c + ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS <= rhs_rows; c += ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS)
            {
                const float16_t *rhs_block = rhs + (size_t)c * rhs_cols;
                float16x8_t vacc = bias ? vld1q(bias + c) : vdupq_n_f16((float16_t)0.0f);

                const float16_t *pl = lhs_row;
                const float16_t *pr = rhs_block;
                vacc = arm_convolve_1_x_n_gather8_f16(&pl, &pr, offsets, first_n, vacc);
                if (rhs_cols > first_n)
                {
                    float32x4_t acc_even = arm_nn_vcvtbq_f32_f16(vacc);
                    float32x4_t acc_odd = arm_nn_vcvttq_f32_f16(vacc);
                    for (int32_t rem = rhs_cols - first_n; rem > 0;)
                    {
                        const int32_t n = (rem > block) ? block : rem;
                        rem -= n;
                        vacc = arm_convolve_1_x_n_gather8_f16(&pl, &pr, offsets, n, vdupq_n_f16((float16_t)0.0f));
                        arm_nn_f16_fold_lanes_f32(&acc_even, &acc_odd, vacc, false);
                    }
                    vacc = arm_nn_f16_narrow_lanes_f32(acc_even, acc_odd);
                }

                vacc = arm_nn_clamp_mve_f16(vacc, vmin, vmax);
                vst1q(dst_row + c, vacc);
            }

            if (rhs_cols <= ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_SUB)
            {
                const mve_pred16_t p = vctp16q((uint32_t)ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS);

                for (; c + ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS <= rhs_rows;
                     c += ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS)
                {
                    const float16_t *rhs_block = rhs + (size_t)c * rhs_cols;
                    float16x8_t vacc = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);

                    const float16_t *pl = lhs_row;
                    const float16_t *pr = rhs_block;
                    vacc = arm_convolve_1_x_n_gather4_f16(&pl, &pr, offsets, p, first_n, vacc);
                    if (rhs_cols > first_n)
                    {
                        float32x4_t acc_even = arm_nn_vcvtbq_f32_f16(vacc);
                        float32x4_t acc_odd = arm_nn_vcvttq_f32_f16(vacc);
                        for (int32_t rem = rhs_cols - first_n; rem > 0;)
                        {
                            const int32_t n = (rem > block) ? block : rem;
                            rem -= n;
                            vacc =
                                arm_convolve_1_x_n_gather4_f16(&pl, &pr, offsets, p, n, vdupq_n_f16((float16_t)0.0f));
                            arm_nn_f16_fold_lanes_f32(&acc_even, &acc_odd, vacc, false);
                        }
                        vacc = arm_nn_f16_narrow_lanes_f32(acc_even, acc_odd);
                    }

                    vacc = arm_nn_clamp_mve_f16(vacc, vmin, vmax);
                    vst1q_p(dst_row + c, vacc, p);
                }
            }
        }
    #else
        int32_t c = 0;
    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        c_tail = c;
        if (rhs_cols > block)
        {
            /* Blockwise remainder columns: second pass below. */
            continue;
        }
    #endif

        for (; c < rhs_rows; ++c)
        {
            const float16_t *rhs_row = rhs + (size_t)c * rhs_cols;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
            _Float16 acc = bias ? (_Float16)bias[c] : (_Float16)0.0f;
            acc += (_Float16)arm_convolve_1_x_n_dot_f16(lhs_row, rhs_row, rhs_cols);
    #else
            const float32_t acc32 =
                (bias ? (float32_t)bias[c] : 0.0f) + arm_convolve_1_x_n_dot_f16(lhs_row, rhs_row, rhs_cols);
            _Float16 acc = (_Float16)acc32;
    #endif

            dst_row[c] = (float16_t)arm_nn_clamp_f16h(acc, (_Float16)activation_max, (_Float16)activation_min);
        }
    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* The remainder columns of a blockwise instantiation run as their own loop nest, which keeps the gather
     * kernels' registers out of the dot product loop; every output is computed exactly as it would be inline. */
    if (rhs_cols > block)
    {
        for (int32_t r = 0; r < lhs_rows; ++r)
        {
            const float16_t *lhs_row = lhs + (size_t)r * lhs_cols_offset;
            float16_t *dst_row = dst + (size_t)r * row_address_offset;
            for (int32_t c = c_tail; c < rhs_rows; ++c)
            {
                const _Float16 acc = arm_convolve_1_x_n_dot_fold_f16(
                    lhs_row, rhs + (size_t)c * rhs_cols, bias ? bias + c : NULL, rhs_cols);
                dst_row[c] = (float16_t)arm_nn_clamp_f16h(acc, (_Float16)activation_max, (_Float16)activation_min);
            }
        }
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/* One out-of-line instantiation per `block`, each with its own register allocation. */
static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_1_x_n_mat_mult_nt_t_strided_fold_f16(const float16_t *__RESTRICT lhs,
                                                  const float16_t *__RESTRICT rhs,
                                                  const float16_t *__RESTRICT bias,
                                                  float16_t *__RESTRICT dst,
                                                  int32_t lhs_rows,
                                                  int32_t rhs_rows,
                                                  int32_t rhs_cols,
                                                  int32_t lhs_cols_offset,
                                                  int32_t row_address_offset,
                                                  float16_t activation_min,
                                                  float16_t activation_max)
{
    return arm_convolve_1_x_n_mat_mult_nt_t_strided_body_f16(lhs,
                                                             rhs,
                                                             bias,
                                                             dst,
                                                             lhs_rows,
                                                             rhs_rows,
                                                             rhs_cols,
                                                             lhs_cols_offset,
                                                             row_address_offset,
                                                             activation_min,
                                                             activation_max,
                                                             ARM_NN_F16_ACC_BLOCK);
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_1_x_n_mat_mult_nt_t_strided_acc16_f16(const float16_t *__RESTRICT lhs,
                                                   const float16_t *__RESTRICT rhs,
                                                   const float16_t *__RESTRICT bias,
                                                   float16_t *__RESTRICT dst,
                                                   int32_t lhs_rows,
                                                   int32_t rhs_rows,
                                                   int32_t rhs_cols,
                                                   int32_t lhs_cols_offset,
                                                   int32_t row_address_offset,
                                                   float16_t activation_min,
                                                   float16_t activation_max)
{
    return arm_convolve_1_x_n_mat_mult_nt_t_strided_body_f16(lhs,
                                                             rhs,
                                                             bias,
                                                             dst,
                                                             lhs_rows,
                                                             rhs_rows,
                                                             rhs_cols,
                                                             lhs_cols_offset,
                                                             row_address_offset,
                                                             activation_min,
                                                             activation_max,
                                                             ARM_NN_F16_ACC_BLOCK_NONE);
}

__STATIC_INLINE void arm_convolve_1_x_n_find_regions(const cmsis_nn_conv_params_f16 *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims,
                                                     int32_t *left_pad_num,
                                                     int32_t *no_pad_num,
                                                     int32_t *right_pad_num)
{
    int32_t first_valid = 0;
    while (first_valid < output_dims->w)
    {
        const int32_t base_x = first_valid * conv_params->stride.w - conv_params->padding.w;
        if (base_x >= 0 && (base_x + filter_dims->w) <= input_dims->w)
        {
            break;
        }
        ++first_valid;
    }

    int32_t last_valid = output_dims->w - 1;
    while (last_valid >= first_valid)
    {
        const int32_t base_x = last_valid * conv_params->stride.w - conv_params->padding.w;
        if (base_x >= 0 && (base_x + filter_dims->w) <= input_dims->w)
        {
            break;
        }
        --last_valid;
    }

    *left_pad_num = first_valid;
    *no_pad_num = ARM_NN_MAX(last_valid - first_valid + 1, 0);
    *right_pad_num = output_dims->w - *left_pad_num - *no_pad_num;
}

/* Route one packed patch tile through the matmul that matches the filter storage format, exactly as the
 * patch-GEMM path in arm_convolve_f16.c does; the 1xN path previously always took the OHWI kernel and
 * silently misread NT_N_PACKED filters. */
__STATIC_FORCEINLINE arm_cmsis_nn_status arm_convolve_1_x_n_mat_mul_f16(const float16_t *lhs,
                                                                        const float16_t *rhs,
                                                                        const float16_t *bias,
                                                                        float16_t *dst,
                                                                        int32_t lhs_rows,
                                                                        int32_t rhs_rows,
                                                                        int32_t rhs_cols,
                                                                        int32_t row_address_offset,
                                                                        const cmsis_nn_conv_params_f16 *conv_params,
                                                                        const bool acc16,
                                                                        const int32_t packed)
{
    if (ARM_CONV_FORMAT_PACKED(packed, conv_params))
    {
        return (acc16 ? arm_nn_mat_mult_nt_n_packed_f16_acc16
                      : arm_nn_mat_mult_nt_n_packed_f16)(lhs,
                                                         rhs,
                                                         bias,
                                                         dst,
                                                         lhs_rows,
                                                         rhs_rows,
                                                         rhs_cols,
                                                         row_address_offset,
                                                         conv_params->activation.min,
                                                         conv_params->activation.max);
    }

    return (acc16 ? arm_nn_mat_mult_nt_t_f16_acc16 : arm_nn_mat_mult_nt_t_f16)(lhs,
                                                                               rhs,
                                                                               bias,
                                                                               dst,
                                                                               lhs_rows,
                                                                               rhs_rows,
                                                                               rhs_cols,
                                                                               row_address_offset,
                                                                               conv_params->activation.min,
                                                                               conv_params->activation.max);
}

static __attribute__((noinline)) void arm_convolve_1_x_n_pack_rows_f16(float16_t *scratch,
                                                                       const float16_t *input_b,
                                                                       const cmsis_nn_conv_params_f16 *conv_params,
                                                                       const cmsis_nn_dims *input_dims,
                                                                       const cmsis_nn_dims *filter_dims,
                                                                       int32_t start_out_x,
                                                                       int32_t rows)
{
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t kernel_w = filter_dims->w;
    const int32_t rhs_cols = kernel_w * input_c;

    for (int32_t r = 0; r < rows; ++r)
    {
        const int32_t out_x = start_out_x + r;
        const int32_t base_x = out_x * conv_params->stride.w - conv_params->padding.w;
        /* Clipped to the kernel width: with padding.w > kernel_w a fully padded position would otherwise
         * zero-fill more than one patch row and run past the scratch buffer. */
        const int32_t left_pad_cols = ARM_NN_MIN(kernel_w, ARM_NN_MAX(0, -base_x));
        const int32_t valid_x0 = ARM_NN_MAX(base_x, 0);
        const int32_t valid_x1 = ARM_NN_MIN(base_x + kernel_w, input_w);
        const int32_t valid_cols = ARM_NN_MAX(valid_x1 - valid_x0, 0);
        const int32_t right_pad_cols = kernel_w - left_pad_cols - valid_cols;
        float16_t *patch_row = scratch + (size_t)r * rhs_cols;

        if (left_pad_cols > 0)
        {
            arm_memset_f16(patch_row, (float16_t)0.0f, (uint32_t)((size_t)left_pad_cols * input_c));
        }
        if (valid_cols > 0)
        {
            arm_memcpy_f16(patch_row + (size_t)left_pad_cols * input_c,
                           input_b + (size_t)valid_x0 * input_c,
                           (uint32_t)((size_t)valid_cols * input_c));
        }
        if (right_pad_cols > 0)
        {
            arm_memset_f16(patch_row + (size_t)(left_pad_cols + valid_cols) * input_c,
                           (float16_t)0.0f,
                           (uint32_t)((size_t)right_pad_cols * input_c));
        }
    }
}

/* Whether the no-padding rows are cheaper one at a time, read in place, through the contiguous-K matmul than through
 * the strided kernel, which puts output channels on the 8 float16 lanes. That holds when the last lane block keeps 4
 * to 7 channels and the reduction has at least 80 taps, or when it has at least 224. MVE builds only. */
__STATIC_FORCEINLINE bool arm_convolve_1_x_n_rows_in_place_f16(const int32_t output_c, const int32_t rhs_cols)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    return output_c >= 4 && ((output_c % 8 >= 4 && rhs_cols >= 80) || rhs_cols >= 224);
    #else
    (void)output_c;
    (void)rhs_cols;
    return false;
    #endif
}

/* Shared body; `block` is ARM_NN_F16_ACC_BLOCK or ARM_NN_F16_ACC_BLOCK_NONE at every call site. */
__STATIC_FORCEINLINE arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f16_body(const cmsis_nn_context *ctx,
                                                                          const cmsis_nn_conv_params_f16 *conv_params,
                                                                          const cmsis_nn_dims *input_dims,
                                                                          const float16_t *input_data,
                                                                          const cmsis_nn_dims *filter_dims,
                                                                          const float16_t *filter_data,
                                                                          const float16_t *bias_data,
                                                                          const cmsis_nn_dims *output_dims,
                                                                          float16_t *output_data,
                                                                          const int32_t block,
                                                                          const int32_t packed)
{
    const bool acc16 = block == ARM_NN_F16_ACC_BLOCK_NONE;

    if (!ctx || !ctx->buf || !conv_params || !input_dims || !input_data || !filter_dims || !filter_data ||
        !output_dims || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (!arm_nn_conv_flt_is_1xn(&conv_params->stride,
                                &conv_params->padding,
                                &conv_params->dilation,
                                input_dims,
                                filter_dims,
                                output_dims) ||
        input_dims->c != filter_dims->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (packed != ARM_CONV_FORMAT_FROM_PARAMS &&
        (conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED) != (packed != 0))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    const int32_t buf_size =
        arm_convolve_1_x_n_f16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
    if (buf_size <= 0 || ctx->size < buf_size)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_w = filter_dims->w;
    const int32_t rhs_cols = kernel_w * input_c;
    const int32_t lhs_cols_offset = input_c * conv_params->stride.w;
    const int32_t tile_rows = (int32_t)((size_t)ctx->size / ((size_t)rhs_cols * sizeof(float16_t)));
    float16_t *scratch = (float16_t *)ctx->buf;
    int32_t left_pad_num = 0;
    int32_t no_pad_num = 0;
    int32_t right_pad_num = 0;

    if (tile_rows <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    arm_convolve_1_x_n_find_regions(
        conv_params, input_dims, filter_dims, output_dims, &left_pad_num, &no_pad_num, &right_pad_num);

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_w * output_c;

        for (int32_t row = 0; row < left_pad_num; row += tile_rows)
        {
            const int32_t rows = ARM_NN_MIN(tile_rows, left_pad_num - row);
            arm_convolve_1_x_n_pack_rows_f16(scratch, input_b, conv_params, input_dims, filter_dims, row, rows);

            arm_cmsis_nn_status st = arm_convolve_1_x_n_mat_mul_f16(scratch,
                                                                    filter_data,
                                                                    bias_data,
                                                                    output_b,
                                                                    rows,
                                                                    output_c,
                                                                    rhs_cols,
                                                                    output_c,
                                                                    conv_params,
                                                                    acc16,
                                                                    packed);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
            output_b += (size_t)rows * output_c;
        }

        if (no_pad_num > 0 && ARM_CONV_FORMAT_PACKED(packed, conv_params))
        {
            /* The no-padding paths below read OHWI filters straight from the input; packed filters take the
             * same pack-rows tile loop as the padded regions so the format-aware matmul can consume them. */
            for (int32_t row = 0; row < no_pad_num; row += tile_rows)
            {
                const int32_t rows = ARM_NN_MIN(tile_rows, no_pad_num - row);
                arm_convolve_1_x_n_pack_rows_f16(
                    scratch, input_b, conv_params, input_dims, filter_dims, left_pad_num + row, rows);

                arm_cmsis_nn_status st = arm_convolve_1_x_n_mat_mul_f16(scratch,
                                                                        filter_data,
                                                                        bias_data,
                                                                        output_b,
                                                                        rows,
                                                                        output_c,
                                                                        rhs_cols,
                                                                        output_c,
                                                                        conv_params,
                                                                        acc16,
                                                                        packed);
                if (st != ARM_CMSIS_NN_SUCCESS)
                {
                    return st;
                }
                output_b += (size_t)rows * output_c;
            }
        }
        else if (no_pad_num > 0 && arm_convolve_1_x_n_rows_in_place_f16(output_c, rhs_cols))
        {
            const float16_t *lhs = input_b + (conv_params->stride.w * left_pad_num - conv_params->padding.w) * input_c;
            for (int32_t row = 0; row < no_pad_num; ++row)
            {
                arm_cmsis_nn_status st = arm_convolve_1_x_n_mat_mul_f16(
                    lhs, filter_data, bias_data, output_b, 1, output_c, rhs_cols, output_c, conv_params, acc16, packed);
                if (st != ARM_CMSIS_NN_SUCCESS)
                {
                    return st;
                }
                lhs += lhs_cols_offset;
                output_b += output_c;
            }
        }
        else if (no_pad_num > 0)
        {
            const int32_t input_start = (conv_params->stride.w * left_pad_num - conv_params->padding.w) * input_c;
            arm_cmsis_nn_status st =
                (acc16 ? arm_convolve_1_x_n_mat_mult_nt_t_strided_acc16_f16
                       : arm_convolve_1_x_n_mat_mult_nt_t_strided_fold_f16)(input_b + input_start,
                                                                            filter_data,
                                                                            bias_data,
                                                                            output_b,
                                                                            no_pad_num,
                                                                            output_c,
                                                                            rhs_cols,
                                                                            lhs_cols_offset,
                                                                            output_c,
                                                                            conv_params->activation.min,
                                                                            conv_params->activation.max);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
            output_b += (size_t)no_pad_num * output_c;
        }

        for (int32_t row = 0; row < right_pad_num; row += tile_rows)
        {
            const int32_t rows = ARM_NN_MIN(tile_rows, right_pad_num - row);
            const int32_t start_out_x = left_pad_num + no_pad_num + row;
            arm_convolve_1_x_n_pack_rows_f16(scratch, input_b, conv_params, input_dims, filter_dims, start_out_x, rows);

            arm_cmsis_nn_status st = arm_convolve_1_x_n_mat_mul_f16(scratch,
                                                                    filter_data,
                                                                    bias_data,
                                                                    output_b,
                                                                    rows,
                                                                    output_c,
                                                                    rhs_cols,
                                                                    output_c,
                                                                    conv_params,
                                                                    acc16,
                                                                    packed);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
            output_b += (size_t)rows * output_c;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/* One out-of-line instantiation per block length, so the public entries below stay thin. */
static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_1_x_n_nhwc_f16_fold(const cmsis_nn_context *ctx,
                                 const cmsis_nn_conv_params_f16 *conv_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float16_t *input_data,
                                 const cmsis_nn_dims *filter_dims,
                                 const float16_t *filter_data,
                                 const float16_t *bias_data,
                                 const cmsis_nn_dims *output_dims,
                                 float16_t *output_data)
{
    return arm_convolve_1_x_n_nhwc_f16_body(ctx,
                                            conv_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_data,
                                            output_dims,
                                            output_data,
                                            ARM_NN_F16_ACC_BLOCK,
                                            ARM_CONV_FORMAT_FROM_PARAMS);
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_1_x_n_nhwc_f16_acc16_impl(const cmsis_nn_context *ctx,
                                       const cmsis_nn_conv_params_f16 *conv_params,
                                       const cmsis_nn_dims *input_dims,
                                       const float16_t *input_data,
                                       const cmsis_nn_dims *filter_dims,
                                       const float16_t *filter_data,
                                       const float16_t *bias_data,
                                       const cmsis_nn_dims *output_dims,
                                       float16_t *output_data)
{
    return arm_convolve_1_x_n_nhwc_f16_body(ctx,
                                            conv_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_data,
                                            output_dims,
                                            output_data,
                                            ARM_NN_F16_ACC_BLOCK_NONE,
                                            ARM_CONV_FORMAT_FROM_PARAMS);
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_conv_params_f16 *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output_data)
{
    (void)bias_dims;
    return arm_convolve_1_x_n_nhwc_f16_fold(
        ctx, conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f16_acc16(const cmsis_nn_context *ctx,
                                                      const cmsis_nn_conv_params_f16 *conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const float16_t *input_data,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const float16_t *filter_data,
                                                      const cmsis_nn_dims *bias_dims,
                                                      const float16_t *bias_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t *output_data)
{
    (void)bias_dims;
    return arm_convolve_1_x_n_nhwc_f16_acc16_impl(
        ctx, conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
}

arm_cmsis_nn_status arm_convolve_1_x_n_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_conv_params_f16 *conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output_data,
                                           arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_convolve_1_x_n_nhwc_f16(ctx,
                                       conv_params,
                                       input_dims,
                                       input_data,
                                       filter_dims,
                                       filter_data,
                                       bias_dims,
                                       bias_data,
                                       output_dims,
                                       output_data);
}

arm_cmsis_nn_status arm_convolve_1_x_n_f16_acc16(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_conv_params_f16 *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float16_t *input_data,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float16_t *filter_data,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float16_t *bias_data,
                                                 const cmsis_nn_dims *output_dims,
                                                 float16_t *output_data,
                                                 arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_convolve_1_x_n_nhwc_f16_acc16(ctx,
                                             conv_params,
                                             input_dims,
                                             input_data,
                                             filter_dims,
                                             filter_data,
                                             bias_dims,
                                             bias_data,
                                             output_dims,
                                             output_data);
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_ohwi_f16(const cmsis_nn_context *ctx,
                                                     const cmsis_nn_conv_params_f16 *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const float16_t *input_data,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const float16_t *filter_data,
                                                     const cmsis_nn_dims *bias_dims,
                                                     const float16_t *bias_data,
                                                     const cmsis_nn_dims *output_dims,
                                                     float16_t *output_data)
{
    (void)bias_dims;
    return arm_convolve_1_x_n_nhwc_f16_body(ctx,
                                            conv_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_data,
                                            output_dims,
                                            output_data,
                                            ARM_NN_F16_ACC_BLOCK,
                                            0);
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_ohwi_f16_acc16(const cmsis_nn_context *ctx,
                                                           const cmsis_nn_conv_params_f16 *conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float16_t *input_data,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const float16_t *filter_data,
                                                           const cmsis_nn_dims *bias_dims,
                                                           const float16_t *bias_data,
                                                           const cmsis_nn_dims *output_dims,
                                                           float16_t *output_data)
{
    (void)bias_dims;
    return arm_convolve_1_x_n_nhwc_f16_body(ctx,
                                            conv_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_data,
                                            output_dims,
                                            output_data,
                                            ARM_NN_F16_ACC_BLOCK_NONE,
                                            0);
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_packed_f16(const cmsis_nn_context *ctx,
                                                       const cmsis_nn_conv_params_f16 *conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const float16_t *input_data,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const float16_t *filter_data,
                                                       const cmsis_nn_dims *bias_dims,
                                                       const float16_t *bias_data,
                                                       const cmsis_nn_dims *output_dims,
                                                       float16_t *output_data)
{
    (void)bias_dims;
    return arm_convolve_1_x_n_nhwc_f16_body(ctx,
                                            conv_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_data,
                                            output_dims,
                                            output_data,
                                            ARM_NN_F16_ACC_BLOCK,
                                            1);
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_packed_f16_acc16(const cmsis_nn_context *ctx,
                                                             const cmsis_nn_conv_params_f16 *conv_params,
                                                             const cmsis_nn_dims *input_dims,
                                                             const float16_t *input_data,
                                                             const cmsis_nn_dims *filter_dims,
                                                             const float16_t *filter_data,
                                                             const cmsis_nn_dims *bias_dims,
                                                             const float16_t *bias_data,
                                                             const cmsis_nn_dims *output_dims,
                                                             float16_t *output_data)
{
    (void)bias_dims;
    return arm_convolve_1_x_n_nhwc_f16_body(ctx,
                                            conv_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_data,
                                            output_dims,
                                            output_data,
                                            ARM_NN_F16_ACC_BLOCK_NONE,
                                            1);
}

/** @} end of NNConv group */

#endif /* ARM_NN_ENABLE_F16 */
