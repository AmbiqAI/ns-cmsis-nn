/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_add_s8
 * Description:  Elementwise add w/ support for broadcasting and scalar
 *
 * $Date:        23 May 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; the quantization parameters and the output
 * requantization parameters are locals of the enclosing function. SCALAR_1 broadcasts one element
 * of input 1 against a run of input 2, SCALAR_2 one element of input 2 against a run of input 1. */
#define ARM_ADD_S8_FULL(a, b, o, n)                                                                                    \
    arm_elementwise_add_s8((a),                                                                                        \
                           (b),                                                                                        \
                           input1_offset,                                                                              \
                           input1_mult,                                                                                \
                           input1_shift,                                                                               \
                           input2_offset,                                                                              \
                           input2_mult,                                                                                \
                           input2_shift,                                                                               \
                           left_shift,                                                                                 \
                           (o),                                                                                        \
                           out_offset,                                                                                 \
                           out_mult,                                                                                   \
                           out_shift,                                                                                  \
                           out_activation_min,                                                                         \
                           out_activation_max,                                                                         \
                           (n))
#define ARM_ADD_S8_SCALAR_1(s, v, o, n)                                                                                \
    arm_add_scalar_s8((s),                                                                                             \
                      (v),                                                                                             \
                      input1_offset,                                                                                   \
                      input1_mult,                                                                                     \
                      input1_shift,                                                                                    \
                      input2_offset,                                                                                   \
                      input2_mult,                                                                                     \
                      input2_shift,                                                                                    \
                      left_shift,                                                                                      \
                      (o),                                                                                             \
                      out_offset,                                                                                      \
                      out_mult,                                                                                        \
                      out_shift,                                                                                       \
                      out_activation_min,                                                                              \
                      out_activation_max,                                                                              \
                      (n))
#define ARM_ADD_S8_SCALAR_2(s, v, o, n)                                                                                \
    arm_add_scalar_s8((s),                                                                                             \
                      (v),                                                                                             \
                      input2_offset,                                                                                   \
                      input2_mult,                                                                                     \
                      input2_shift,                                                                                    \
                      input1_offset,                                                                                   \
                      input1_mult,                                                                                     \
                      input1_shift,                                                                                    \
                      left_shift,                                                                                      \
                      (o),                                                                                             \
                      out_offset,                                                                                      \
                      out_mult,                                                                                        \
                      out_shift,                                                                                       \
                      out_activation_min,                                                                              \
                      out_activation_max,                                                                              \
                      (n))

#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
/*
 * Add the single row `row` (c elements) to n_rows consecutive rows of c elements of vec, with the arithmetic
 * of arm_elementwise_add_s8. The row operand is requantized once per block of 4 channels with
 * arm_requantize_mve; for the vec operand and the output, the zero left shift of arm_requantize_mve is
 * dropped, and so is the divide when vec_shift is 0 (vec_div, a compile-time constant at each call site).
 * The rounding divide for a non-zero exponent e is taken as vrshl(x + (x >> 31), -e), which is what
 * arm_divide_by_power_of_two_mve computes for e != 0.
 * Preconditions: vec_shift and row_shift in [-31, 0], left_shift in [0, 31], out_shift in [-31, -1].
 */
__STATIC_FORCEINLINE void arm_add_s8_row_broadcast(const int8_t *vec,
                                                   const int8_t *row,
                                                   const int32_t vec_offset,
                                                   const int32_t vec_mult,
                                                   const int32_t vec_shift,
                                                   const int32_t row_offset,
                                                   const int32_t row_mult,
                                                   const int32_t row_shift,
                                                   const int32_t left_shift,
                                                   int8_t *output,
                                                   const int32_t out_offset,
                                                   const int32_t out_mult,
                                                   const int32_t out_shift,
                                                   const int32_t out_activation_min,
                                                   const int32_t out_activation_max,
                                                   const int32_t n_rows,
                                                   const int32_t c,
                                                   const int vec_div)
{
    const int32x4_t vec_neg_exp = vdupq_n_s32(vec_shift);
    const int32x4_t out_neg_exp = vdupq_n_s32(out_shift);
    const int32x4_t act_min = vdupq_n_s32(out_activation_min);
    const int32x4_t act_max = vdupq_n_s32(out_activation_max);

    for (int32_t c0 = 0; c0 < c; c0 += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(c - c0));
        int32x4_t r = vldrbq_z_s32(row + c0, p);
        r = vshlq_r_s32(vaddq_n_s32(r, row_offset), left_shift);
        r = arm_requantize_mve(r, row_mult, row_shift);

        const int8_t *v_ptr = vec + c0;
        int8_t *o_ptr = output + c0;

        for (int32_t i = 0; i < n_rows; i++)
        {
            int32x4_t v = vldrbq_z_s32(v_ptr, p);
            v = vqrdmulhq_n_s32(vshlq_r_s32(vaddq_n_s32(v, vec_offset), left_shift), vec_mult);
            if (vec_div)
            {
                v = arm_divide_by_nonzero_power_of_two_mve(v, vec_neg_exp);
            }
            v = vqrdmulhq_n_s32(vaddq_s32(v, r), out_mult);
            v = arm_divide_by_nonzero_power_of_two_mve(v, out_neg_exp);
            v = vaddq_n_s32(v, out_offset);
            v = vmaxq_s32(v, act_min);
            v = vminq_s32(v, act_max);

            vstrbq_p_s32(o_ptr, v, p);

            v_ptr += c;
            o_ptr += c;
        }
    }
}
#endif

/*
 * s8 elementwise add w/ support for broadcasting and scalar
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_add_s8(const int8_t *input1_data,
                               const cmsis_nn_dims *input1_dims,
                               const int8_t *input2_data,
                               const cmsis_nn_dims *input2_dims,
                               const int32_t input1_offset,
                               const int32_t input1_mult,
                               const int32_t input1_shift,
                               const int32_t input2_offset,
                               const int32_t input2_mult,
                               const int32_t input2_shift,
                               const int32_t left_shift,
                               int8_t *output_data,
                               const cmsis_nn_dims *output_dims,
                               const int32_t out_offset,
                               const int32_t out_mult,
                               const int32_t out_shift,
                               const int32_t out_activation_min,
                               const int32_t out_activation_max)
{
    if (!input1_data || !input2_data || !output_data || !input1_dims || !input2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input1_dims, input2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    /* One operand broadcast along W with matching C (e.g. [N,H,W,C] + [N|1,H|1,1,C]): the walk would call
     * arm_elementwise_add_s8 once per C-element row; run the rows in one pass instead. */
    if (input1_dims->c == input2_dims->c && input1_dims->c > 1 && (input1_dims->w == 1) != (input2_dims->w == 1) &&
        output_dims->w > 1 && input1_shift <= 0 && input1_shift >= -31 && input2_shift <= 0 && input2_shift >= -31 &&
        left_shift >= 0 && left_shift <= 31 && out_shift < 0 && out_shift >= -31)
    {
        const int32_t vec_is_1 = input2_dims->w == 1;
        const int8_t *vec = vec_is_1 ? input1_data : input2_data;
        const int8_t *row = vec_is_1 ? input2_data : input1_data;
        const cmsis_nn_dims *vec_dims = vec_is_1 ? input1_dims : input2_dims;
        const cmsis_nn_dims *row_dims = vec_is_1 ? input2_dims : input1_dims;
        const int32_t vec_offset = vec_is_1 ? input1_offset : input2_offset;
        const int32_t vec_mult = vec_is_1 ? input1_mult : input2_mult;
        const int32_t vec_shift = vec_is_1 ? input1_shift : input2_shift;
        const int32_t row_offset = vec_is_1 ? input2_offset : input1_offset;
        const int32_t row_mult = vec_is_1 ? input2_mult : input1_mult;
        const int32_t row_shift = vec_is_1 ? input2_shift : input1_shift;
        const int32_t c = output_dims->c;
        const int32_t w = output_dims->w;
        const int32_t vec_n_stride = (vec_dims->n == 1) ? 0 : vec_dims->h * w * c;
        const int32_t vec_h_stride = (vec_dims->h == 1) ? 0 : w * c;
        const int32_t row_n_stride = (row_dims->n == 1) ? 0 : row_dims->h * c;
        const int32_t row_h_stride = (row_dims->h == 1) ? 0 : c;

        for (int32_t n = 0; n < output_dims->n; n++)
        {
            for (int32_t h = 0; h < output_dims->h; h++)
            {
                const int8_t *vec_nh = vec + n * vec_n_stride + h * vec_h_stride;
                const int8_t *row_nh = row + n * row_n_stride + h * row_h_stride;
                if (vec_shift == 0)
                {
                    arm_add_s8_row_broadcast(vec_nh,
                                             row_nh,
                                             vec_offset,
                                             vec_mult,
                                             vec_shift,
                                             row_offset,
                                             row_mult,
                                             row_shift,
                                             left_shift,
                                             output_data,
                                             out_offset,
                                             out_mult,
                                             out_shift,
                                             out_activation_min,
                                             out_activation_max,
                                             w,
                                             c,
                                             0);
                }
                else
                {
                    arm_add_s8_row_broadcast(vec_nh,
                                             row_nh,
                                             vec_offset,
                                             vec_mult,
                                             vec_shift,
                                             row_offset,
                                             row_mult,
                                             row_shift,
                                             left_shift,
                                             output_data,
                                             out_offset,
                                             out_mult,
                                             out_shift,
                                             out_activation_min,
                                             out_activation_max,
                                             w,
                                             c,
                                             1);
                }
                output_data += w * c;
            }
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
#endif

    ARM_NN_BROADCAST_WALK_NHWC(int8_t,
                               int8_t,
                               input1_data,
                               input1_dims,
                               input2_data,
                               input2_dims,
                               output_data,
                               output_dims,
                               ARM_ADD_S8_FULL,
                               ARM_ADD_S8_SCALAR_1,
                               ARM_ADD_S8_SCALAR_2);

    return ARM_CMSIS_NN_SUCCESS;
}

#undef ARM_ADD_S8_FULL
#undef ARM_ADD_S8_SCALAR_1
#undef ARM_ADD_S8_SCALAR_2

/**
 * @} end of Doxygen group
 */
