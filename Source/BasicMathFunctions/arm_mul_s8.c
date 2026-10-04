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
 * Title:        arm_mul_s8
 * Description:  Elementwise mul w/ support for broadcasting and scalar
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
#define ARM_MUL_S8_FULL(a, b, o, n)                                                                                    \
    arm_elementwise_mul_s8((a),                                                                                        \
                           (b),                                                                                        \
                           input1_offset,                                                                              \
                           input2_offset,                                                                              \
                           (o),                                                                                        \
                           out_offset,                                                                                 \
                           out_mult,                                                                                   \
                           out_shift,                                                                                  \
                           out_activation_min,                                                                         \
                           out_activation_max,                                                                         \
                           (n))
#define ARM_MUL_S8_SCALAR_1(s, v, o, n)                                                                                \
    arm_mul_scalar_s8((s),                                                                                             \
                      (v),                                                                                             \
                      input1_offset,                                                                                   \
                      input2_offset,                                                                                   \
                      (o),                                                                                             \
                      out_offset,                                                                                      \
                      out_mult,                                                                                        \
                      out_shift,                                                                                       \
                      out_activation_min,                                                                              \
                      out_activation_max,                                                                              \
                      (n))
#define ARM_MUL_S8_SCALAR_2(s, v, o, n)                                                                                \
    arm_mul_scalar_s8((s),                                                                                             \
                      (v),                                                                                             \
                      input2_offset,                                                                                   \
                      input1_offset,                                                                                   \
                      (o),                                                                                             \
                      out_offset,                                                                                      \
                      out_mult,                                                                                        \
                      out_shift,                                                                                       \
                      out_activation_min,                                                                              \
                      out_activation_max,                                                                              \
                      (n))

#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
/*
 * Multiply n_rows consecutive rows of c elements of vec by the single row `row` (c elements), with the
 * arithmetic of arm_elementwise_mul_s8: 16-bit offset add, 16x16->32 product, arm_requantize_mve_32x4,
 * saturating narrow to 16 bits, output offset and clamp. The output offset add saturates, so a
 * requantized product outside the int16 range clamps like the scalar reference instead of wrapping.
 * The channels are walked in blocks of 8 so the row block and all constants stay in registers across
 * the rows. neg_shift is a compile-time constant
 * at each call site: for out_shift < 0 the left shift is 0 and the divide exponent is non-zero, so the
 * shift and the zero-exponent mask of arm_divide_by_power_of_two_mve_32x4 drop out; otherwise the
 * divide is the identity.
 */
__STATIC_FORCEINLINE void arm_mul_s8_row_broadcast(const int8_t *vec,
                                                   const int8_t *row,
                                                   const int32_t vec_offset,
                                                   const int32_t row_offset,
                                                   int8_t *output,
                                                   const int32_t out_offset,
                                                   const int32_t out_mult,
                                                   const int32_t out_shift,
                                                   const int32_t out_activation_min,
                                                   const int32_t out_activation_max,
                                                   const int32_t n_rows,
                                                   const int32_t c,
                                                   const int neg_shift)
{
    const int32x4_t shift = vdupq_n_s32(out_shift);
    const int16x8_t act_min = vdupq_n_s16((int16_t)out_activation_min);
    const int16x8_t act_max = vdupq_n_s16((int16_t)out_activation_max);

    for (int32_t c0 = 0; c0 < c; c0 += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(c - c0));
        const int16x8_t r = vaddq_n_s16(vldrbq_z_s16(row + c0, p), (int16_t)row_offset);
        const int8_t *v_ptr = vec + c0;
        int8_t *o_ptr = output + c0;

        for (int32_t i = 0; i < n_rows; i++)
        {
            const int16x8_t v = vaddq_n_s16(vldrbq_z_s16(v_ptr, p), (int16_t)vec_offset);
            int32x4_t res_a = vmullbq_int_s16(v, r);
            int32x4_t res_b = vmulltq_int_s16(v, r);

            if (neg_shift)
            {
                res_a = vqrdmulhq_n_s32(res_a, out_mult);
                res_b = vqrdmulhq_n_s32(res_b, out_mult);
                res_a = arm_divide_by_nonzero_power_of_two_mve(res_a, shift);
                res_b = arm_divide_by_nonzero_power_of_two_mve(res_b, shift);
            }
            else
            {
                res_a = vqrdmulhq_n_s32(vshlq_s32(res_a, shift), out_mult);
                res_b = vqrdmulhq_n_s32(vshlq_s32(res_b, shift), out_mult);
            }

            int16x8_t res = vqmovntq_s32(vqmovnbq_s32(vdupq_n_s16(0), res_a), res_b);
            res = vqaddq_n_s16(res, (int16_t)out_offset);
            res = vmaxq_s16(res, act_min);
            res = vminq_s16(res, act_max);

            vstrbq_p_s16(o_ptr, res, p);

            v_ptr += c;
            o_ptr += c;
        }
    }
}
#endif

#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
/* The row-broadcast route in one pass over the rows (MVE, default rounding) */
__STATIC_FORCEINLINE void arm_mul_s8_row_broadcast_rows(const int8_t *input1_data,
                                                        const cmsis_nn_dims *input1_dims,
                                                        const int8_t *input2_data,
                                                        const cmsis_nn_dims *input2_dims,
                                                        const int32_t input1_offset,
                                                        const int32_t input2_offset,
                                                        int8_t *output_data,
                                                        const cmsis_nn_dims *output_dims,
                                                        const int32_t out_offset,
                                                        const int32_t out_mult,
                                                        const int32_t out_shift,
                                                        const int32_t out_activation_min,
                                                        const int32_t out_activation_max)
{
    const int32_t vec_is_1 = input2_dims->w == 1;
    const int8_t *vec = vec_is_1 ? input1_data : input2_data;
    const int8_t *row = vec_is_1 ? input2_data : input1_data;
    const cmsis_nn_dims *vec_dims = vec_is_1 ? input1_dims : input2_dims;
    const cmsis_nn_dims *row_dims = vec_is_1 ? input2_dims : input1_dims;
    const int32_t vec_offset = vec_is_1 ? input1_offset : input2_offset;
    const int32_t row_offset = vec_is_1 ? input2_offset : input1_offset;
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
            if (out_shift < 0)
            {
                arm_mul_s8_row_broadcast(vec_nh,
                                         row_nh,
                                         vec_offset,
                                         row_offset,
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
            else
            {
                arm_mul_s8_row_broadcast(vec_nh,
                                         row_nh,
                                         vec_offset,
                                         row_offset,
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
            output_data += w * c;
        }
    }
}
#else

/* The row-broadcast route as one arm_elementwise_mul_s8() call per output pixel: the calls the broadcast walk makes
 * for this shape */
static void arm_mul_s8_row_broadcast_pixels(const int8_t *input1_data,
                                            const cmsis_nn_dims *input1_dims,
                                            const int8_t *input2_data,
                                            const cmsis_nn_dims *input2_dims,
                                            const int32_t input1_offset,
                                            const int32_t input2_offset,
                                            int8_t *output_data,
                                            const cmsis_nn_dims *output_dims,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max)
{
    const int32_t c = output_dims->c;
    const int32_t w1 = input1_dims->w == 1 ? 0 : c;
    const int32_t w2 = input2_dims->w == 1 ? 0 : c;
    const int32_t h1 = input1_dims->h == 1 ? 0 : input1_dims->w * c;
    const int32_t h2 = input2_dims->h == 1 ? 0 : input2_dims->w * c;
    const int32_t n1 = input1_dims->n == 1 ? 0 : input1_dims->h * input1_dims->w * c;
    const int32_t n2 = input2_dims->n == 1 ? 0 : input2_dims->h * input2_dims->w * c;
    for (int32_t n = 0; n < output_dims->n; n++)
    {
        for (int32_t h = 0; h < output_dims->h; h++)
        {
            for (int32_t x = 0; x < output_dims->w; x++)
            {
                arm_elementwise_mul_s8(input1_data + n * n1 + h * h1 + x * w1,
                                       input2_data + n * n2 + h * h2 + x * w2,
                                       input1_offset,
                                       input2_offset,
                                       output_data,
                                       out_offset,
                                       out_mult,
                                       out_shift,
                                       out_activation_min,
                                       out_activation_max,
                                       c);
                output_data += c;
            }
        }
    }
}
#endif

/*
 * s8 elementwise mul w/ support for broadcasting and scalar
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_mul_s8(const int8_t *input1_data,
                               const cmsis_nn_dims *input1_dims,
                               const int8_t *input2_data,
                               const cmsis_nn_dims *input2_dims,
                               const int32_t input1_offset,
                               const int32_t input2_offset,
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
    /* One operand broadcast along W with matching C (e.g. [N,H,W,C] x [N|1,H|1,1,C], the squeeze-and-excite
     * scale): the walk would call arm_elementwise_mul_s8 once per C-element row; run the rows in one pass. */
    if (arm_nn_is_row_broadcast(input1_dims, input2_dims, output_dims))
    {
        arm_mul_s8_row_broadcast_rows(input1_data,
                                      input1_dims,
                                      input2_data,
                                      input2_dims,
                                      input1_offset,
                                      input2_offset,
                                      output_data,
                                      output_dims,
                                      out_offset,
                                      out_mult,
                                      out_shift,
                                      out_activation_min,
                                      out_activation_max);
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
                               ARM_MUL_S8_FULL,
                               ARM_MUL_S8_SCALAR_1,
                               ARM_MUL_S8_SCALAR_2);

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * s8 elementwise mul on the row-broadcast route.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_mul_row_broadcast_s8(const int8_t *input1_data,
                                             const cmsis_nn_dims *input1_dims,
                                             const int8_t *input2_data,
                                             const cmsis_nn_dims *input2_dims,
                                             const int32_t input1_offset,
                                             const int32_t input2_offset,
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
    if (!arm_nn_is_row_broadcast(input1_dims, input2_dims, output_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    arm_mul_s8_row_broadcast_rows(input1_data,
                                  input1_dims,
                                  input2_data,
                                  input2_dims,
                                  input1_offset,
                                  input2_offset,
                                  output_data,
                                  output_dims,
                                  out_offset,
                                  out_mult,
                                  out_shift,
                                  out_activation_min,
                                  out_activation_max);
#else
    arm_mul_s8_row_broadcast_pixels(input1_data,
                                    input1_dims,
                                    input2_data,
                                    input2_dims,
                                    input1_offset,
                                    input2_offset,
                                    output_data,
                                    output_dims,
                                    out_offset,
                                    out_mult,
                                    out_shift,
                                    out_activation_min,
                                    out_activation_max);
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

#undef ARM_MUL_S8_FULL
#undef ARM_MUL_S8_SCALAR_1
#undef ARM_MUL_S8_SCALAR_2

/**
 * @} end of Doxygen group
 */
