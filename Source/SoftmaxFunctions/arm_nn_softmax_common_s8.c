/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_softmax_common_s8.c
 * Description:  Softmax with s8 input and output of s8 or s16.
 *
 * $Date:        5 January 2023
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#define ACCUM_BITS 12

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && !defined(ARM_GCC_12_2_ICE)
static int32x4_t arm_exp_on_negative_values_mve_32x4(int32x4_t val)
{
    const int32x4_t val_mod_minus_quarter = vsubq_n_s32(vandq_s32(val, vdupq_n_s32((1 << 24) - 1)), 1 << 24);
    const int32x4_t remainder = vsubq_s32(val_mod_minus_quarter, val);
    /* vshlq_n_s32, not <<: the lanes are negative, and a left shift of a negative int32_t is undefined */
    const int32x4_t x = vaddq_n_s32(vshlq_n_s32(val_mod_minus_quarter, 5), 1 << 28);
    const int32x4_t x2 = MUL_SAT_MVE(x, x);
    /* The rounding divisions by 2^2 and 2^1, with their fixup for negative dividends */
    const int32x4_t x4 = MUL_SAT_MVE(x2, x2);
    const int32x4_t op_1 = vaddq_s32(vrshrq_n_s32(vqaddq_s32(x4, vshrq_n_s32(x4, 31)), 2), MUL_SAT_MVE(x2, x));
    const int32x4_t op_3 = vaddq_s32(vqrdmulhq_n_s32(op_1, 715827883), x2);
    const int32x4_t op_2 = vaddq_s32(x, vrshrq_n_s32(vqaddq_s32(op_3, vshrq_n_s32(op_3, 31)), 1));
    int32x4_t result = vaddq_n_s32(vqrdmulhq_n_s32(op_2, 1895147668), 1895147668);

    /* Bit 31 - up of remainder, tested as the sign bit after shifting it up. Bits 24 to 30, in that order. */
    #define MUL_IF_BIT_SET(up, multiplier)                                                                             \
        result = vqrdmulhq_m_n_s32(result, result, (multiplier), vcmpltq_n_s32(vshlq_n_s32(remainder, (up)), 0))

    MUL_IF_BIT_SET(7, 1672461947);
    MUL_IF_BIT_SET(6, 1302514674);
    MUL_IF_BIT_SET(5, 790015084);
    MUL_IF_BIT_SET(4, 290630308);
    MUL_IF_BIT_SET(3, 39332535);
    MUL_IF_BIT_SET(2, 720401);
    MUL_IF_BIT_SET(1, 242);

    #undef MUL_IF_BIT_SET

    return vpselq_s32(vdupq_n_s32(NN_Q31_MAX), result, vcmpeqq_n_s32(val, 0));
}

/* Exponential of the differences -k in the lanes of p where -k >= diff_min, and 0 in all other lanes. A 0
 * contributes nothing to the row sum and gives the minimum output, as an excluded difference does. */
static int32x4_t arm_nn_softmax_exp_mve(const int32x4_t k,
                                        const int32_t mask,
                                        const int32_t mult,
                                        const int32_t diff_min,
                                        const mve_pred16_t p)
{
    const int32x4_t diff = vnegq_s32(k);
    const mve_pred16_t valid = vcmpgeq_m_n_s32(diff, diff_min, p);
    if (valid == 0)
    {
        return vdupq_n_s32(0);
    }
    const int32x4_t res = arm_exp_on_negative_values_mve_32x4(MUL_SAT_MVE(vmulq_n_s32(diff, mask), vdupq_n_s32(mult)));
    return vpselq_s32(res, vdupq_n_s32(0), valid);
}

/* Output of each exponential e: DIV_POW2(MUL_SAT(scale, e), bits) + out_min, clamped to out_max. Both factors are
 * non-negative, so the rounding shift needs no fixup and the result no lower clamp. */
static inline int32x4_t arm_nn_softmax_out_mve(const int32x4_t e,
                                               const int32_t scale,
                                               const int32_t neg_bits,
                                               const int32_t out_min,
                                               const int32x4_t vout_max)
{
    return vminq_s32(vaddq_n_s32(vrshlq_n_s32(vqrdmulhq_n_s32(e, scale), neg_bits), out_min), vout_max);
}

/* arm_nn_one_over_one_plus_x_for_x_in_0_1() per lane, for lanes in [0, NN_Q31_MAX]. MUL_POW2 is a saturating
 * shift, and the halving of val + NN_Q31_MAX rounds to val / 2 + 2^30 for non-negative val. */
static int32x4_t arm_nn_one_over_one_plus_x_mve(const int32x4_t val)
{
    const int32x4_t half_denominator = vaddq_n_s32(vshrq_n_s32(val, 1), 1 << 30);
    int32x4_t x = vaddq_n_s32(vqrdmulhq_n_s32(half_denominator, -1010580540), 1515870810);

    for (int i = 0; i < 3; i++)
    {
        const int32x4_t t = vsubq_s32(vdupq_n_s32(1 << 29), vqrdmulhq_s32(half_denominator, x));
        x = vaddq_s32(x, vqshlq_n_s32(vqrdmulhq_s32(x, t), 2));
    }
    return vqshlq_n_s32(x, 1);
}
#endif

/**
 * @ingroup groupSupport
 */

/**
 * @defgroup supportSoftmax Softmax
 *
 * Support functions for Softmax
 *
 */

/**
 * @addtogroup supportSoftmax
 * @{
 */

/*
 * Softmax function with s8 input and output of s8 or s16.
 *
 * Refer header file for details.
 *
 */
void arm_nn_softmax_common_s8(const int8_t *input,
                              const int32_t num_rows,
                              const int32_t row_size,
                              const int32_t mult,
                              const int32_t shift,
                              const int32_t diff_min,
                              const bool int16_output,
                              void *output)
{
    /* Keep the shift and multiply defined at shift 31, matching the MVE lane arithmetic. */
    const uint32_t mask = 1U << shift;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && !defined(ARM_GCC_12_2_ICE)
    /* exp_table[k] is the exponential of the difference -k, filled on demand. It pays off once the new entries a
     * row needs are no more than the elements still to process. A row that needs more computes the exponentials of its
     * own differences and stores them at their indices. */
    int32_t exp_table[256];
    int32_t table_len = 0;
    int32_t elems_left = num_rows * row_size;
    const int32_t valid_len = diff_min < -256 ? 256 : 1 - diff_min;
    const int32_t unit_bits = ACCUM_BITS + (int16_output ? 15 : 23);
    const int32_t out_min = int16_output ? NN_Q15_MIN : NN_Q7_MIN;
    const int32_t out_max = int16_output ? NN_Q15_MAX : NN_Q7_MAX;

    /* No difference reaches a positive diff_min, so every output is the minimum */
    if (diff_min > 0)
    {
        for (int32_t i = 0; i < elems_left; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(elems_left - i));
            if (int16_output)
            {
                vstrhq_p_s16((int16_t *)output + i, vdupq_n_s16(NN_Q15_MIN), p);
            }
            else
            {
                vstrbq_p_s16((int8_t *)output + i, vdupq_n_s16(NN_Q7_MIN), p);
            }
        }
        return;
    }

    /* Rows awaiting their output pass. Their scales are computed together, as in arm_nn_softmax_row_scale(). */
    int32_t pending = 0;
    int32_t row_max[4];
    int32_t row_bits[4];
    int32_t row_top[4] = {0};
    int32_t row_scale[4];
    bool row_zero[4];

    const int32x4_t vout_max = vdupq_n_s32(out_max);

    for (int32_t row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        int8x16_t vmax = vdupq_n_s8(NN_Q7_MIN);
        int8x16_t vmin = vdupq_n_s8(NN_Q7_MAX);
        for (int32_t col = 0; col < row_size; col += 16)
        {
            const mve_pred16_t p = vctp8q((uint32_t)(row_size - col));
            const int8x16_t ip = vldrbq_z_s8(&input[col], p);
            vmax = vmaxq_m_s8(vmax, vmax, ip, p);
            vmin = vminq_m_s8(vmin, vmin, ip, p);
        }
        const int32_t max = vmaxvq_s8(NN_Q7_MIN, vmax);
        const int32_t range = max - vminvq_s8(NN_Q7_MAX, vmin) + 1;
        /* Lookups skip k >= valid_len, so the row needs only entries below it */
        const int32_t need = ARM_NN_MIN(range, valid_len);

        if (need > table_len && need - table_len <= elems_left)
        {
            for (; table_len < need; table_len += 4)
            {
                const int32x4_t k = vreinterpretq_s32_u32(vidupq_u32((uint32_t)table_len, 1));
                vst1q_s32(&exp_table[table_len], arm_nn_softmax_exp_mve(k, (int32_t)mask, mult, diff_min, 0xFFFF));
            }
        }
        elems_left -= row_size;
        const int32x4_t vmax32 = vdupq_n_s32(max);

        int64_t sum = 0;
        if (need <= table_len)
        {
            for (int32_t col = 0; col < row_size; col += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(row_size - col));
                const int32x4_t k = vsubq_s32(vmax32, vldrbq_z_s32(&input[col], p));
                const mve_pred16_t valid = vcmpltq_m_n_s32(k, valid_len, p);
                const int32x4_t res = vldrwq_gather_shifted_offset_z_s32(exp_table, vreinterpretq_u32_s32(k), valid);
                sum = vaddlvaq_s32(sum, vrshrq_n_s32(res, ACCUM_BITS));
            }
        }
        else
        {
            /* Every table write is the exponential of its index, so the output pass can look these up too */
            for (int32_t col = 0; col < row_size; col += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(row_size - col));
                const int32x4_t k = vsubq_s32(vmax32, vldrbq_z_s32(&input[col], p));
                const int32x4_t res = arm_nn_softmax_exp_mve(k, (int32_t)mask, mult, diff_min, p);
                vstrwq_scatter_shifted_offset_p_s32(exp_table, vreinterpretq_u32_s32(k), res, p);
                sum = vaddlvaq_s32(sum, vrshrq_n_s32(res, ACCUM_BITS));
            }
        }

        /* The scalar part of arm_nn_softmax_row_scale(); the reciprocal is left to the output pass */
        const uint64_t u = (uint64_t)sum;
        const uint32_t high = (uint32_t)(u >> 32);
        const int32_t headroom = high ? (int32_t)CLZ(high) - 32 : (int32_t)CLZ((uint32_t)u);
        const int32_t bits_over_unit = unit_bits - headroom;
        const bool zero = sum <= 0 || bits_over_unit > 31;
        uint32_t top = 0;
        if (!zero)
        {
            top = headroom >= 0 ? (uint32_t)u << headroom : (uint32_t)(u >> -headroom);
        }
        row_max[pending] = max;
        row_bits[pending] = zero ? 31 : bits_over_unit;
        row_top[pending] = (int32_t)(top ^ 0x80000000U);
        row_zero[pending] = zero;
        pending++;
        input += row_size;

        if (pending == 4 || row_idx == num_rows - 1)
        {
            vst1q_s32(row_scale, arm_nn_one_over_one_plus_x_mve(vld1q_s32(row_top)));

            for (int32_t j = 0; j < pending; j++)
            {
                const int8_t *row_in = input - (pending - j) * row_size;
                const int32_t out_offset = (row_idx + 1 - pending + j) * row_size;
                const int32x4_t row_vmax32 = vdupq_n_s32(row_max[j]);
                const int32_t shifted_scale = row_zero[j] ? 0 : row_scale[j];
                const int32_t neg_bits = -row_bits[j];

                for (int32_t col = 0; col < row_size; col += 4)
                {
                    const mve_pred16_t p = vctp32q((uint32_t)(row_size - col));
                    const int32x4_t k = vsubq_s32(row_vmax32, vldrbq_z_s32(&row_in[col], p));
                    const mve_pred16_t valid = vcmpltq_m_n_s32(k, valid_len, p);
                    const int32x4_t res = arm_nn_softmax_out_mve(
                        vldrwq_gather_shifted_offset_z_s32(exp_table, vreinterpretq_u32_s32(k), valid),
                        shifted_scale,
                        neg_bits,
                        out_min,
                        vout_max);
                    if (int16_output)
                    {
                        vstrhq_p_s32((int16_t *)output + out_offset + col, res, p);
                    }
                    else
                    {
                        vstrbq_p_s32((int8_t *)output + out_offset + col, res, p);
                    }
                }
            }
            pending = 0;
        }
    }
#else
    int32_t col = 0;
    int32_t row_idx;

    for (row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        // Find the maximum value in order to ensure numerical stability
        int8_t max = *input;

        for (col = 1; col < row_size; ++col)
        {
            max = ARM_NN_MAX(max, input[col]);
        }

        int32_t diff = 0;
        int64_t sum = 0;

        for (col = 0; col < row_size; ++col)
        {
            diff = input[col] - max;
            if (diff >= diff_min)
            {
                sum += DIV_POW2(EXP_ON_NEG(MUL_SAT((int32_t)((uint32_t)diff * mask), mult)), ACCUM_BITS);
            }
        }

        int32_t bits_over_unit;
        const int32_t shifted_scale =
            arm_nn_softmax_row_scale(sum, ACCUM_BITS + (int16_output ? 15 : 23), &bits_over_unit);

        if (int16_output)
        {
            int16_t *output_s16 = (int16_t *)output + row_idx * row_size;

            for (col = 0; col < row_size; ++col)
            {
                diff = input[col] - max;

                if (diff >= diff_min)
                {
                    const int32_t res =
                        DIV_POW2(MUL_SAT(shifted_scale, EXP_ON_NEG(MUL_SAT((int32_t)((uint32_t)diff * mask), mult))),
                                 bits_over_unit) +
                        NN_Q15_MIN;
                    output_s16[col] = (int16_t)ARM_NN_CLAMP(res, (int32_t)NN_Q15_MAX, (int32_t)NN_Q15_MIN);
                }
                else
                {
                    output_s16[col] = NN_Q15_MIN;
                }
            }
        }
        else
        {
            int8_t *output_s8 = (int8_t *)output + row_idx * row_size;

            for (col = 0; col < row_size; ++col)
            {
                diff = input[col] - max;
                if (diff >= diff_min)
                {
                    const int32_t res =
                        DIV_POW2(MUL_SAT(shifted_scale, EXP_ON_NEG(MUL_SAT((int32_t)((uint32_t)diff * mask), mult))),
                                 bits_over_unit) +
                        NN_Q7_MIN;
                    output_s8[col] = (int8_t)ARM_NN_CLAMP(res, (int32_t)NN_Q7_MAX, (int32_t)NN_Q7_MIN);
                }
                else
                {
                    output_s8[col] = NN_Q7_MIN;
                }
            }
        }

        input += row_size;
    }
#endif
}

/**
 * @} end of Doxygen group
 */
