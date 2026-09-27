/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
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
 * Title:        arm_elementwise_add_s8
 * Description:  Elementwise add
 *
 * $Date:        5 January 2023
 * $Revision:    V.3.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/*
 * s8 elementwise add for two vectors w/ same dimensions
 *
 * Refer header file for details.
 *
 */

/* Note: __SHIFT is expected to be <=0 */

#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
/* Rounding divide by 2^e, half away from zero, for e in [1, 31] given as neg_exp = -e. Equal to
 * arm_divide_by_power_of_two_mve for a non-zero exponent: the sign mask it builds from (x & -e) is
 * the sign of x whenever e != 0, so the AND is dropped. */
__STATIC_FORCEINLINE int32x4_t arm_elementwise_add_s8_rdiv(const int32x4_t x, const int32x4_t neg_exp)
{
    return vrshlq_s32(vqaddq_s32(x, vshrq_n_s32(x, 31)), neg_exp);
}

/* MVE add for input shifts in [-31, 0], left_shift in [0, 31] and out_shift in [-31, -1]: each
 * requantization is arm_requantize_mve with its zero left shift and, for a zero input shift, its
 * identity divide removed. in_1_div/in_2_div are compile-time constants at every call site. */
__STATIC_FORCEINLINE void arm_elementwise_add_s8_mve(const int8_t *input_1_vect,
                                                     const int8_t *input_2_vect,
                                                     const int32_t input_1_offset,
                                                     const int32_t input_1_mult,
                                                     const int32_t input_1_shift,
                                                     const int32_t input_2_offset,
                                                     const int32_t input_2_mult,
                                                     const int32_t input_2_shift,
                                                     const int32_t left_shift,
                                                     int8_t *output,
                                                     const int32_t out_offset,
                                                     const int32_t out_mult,
                                                     const int32_t out_shift,
                                                     const int32_t out_activation_min,
                                                     const int32_t out_activation_max,
                                                     int32_t count,
                                                     const int in_1_div,
                                                     const int in_2_div)
{
    const int32x4_t neg_exp_1 = vdupq_n_s32(input_1_shift);
    const int32x4_t neg_exp_2 = vdupq_n_s32(input_2_shift);
    const int32x4_t neg_exp_out = vdupq_n_s32(out_shift);
    const int32x4_t act_min = vdupq_n_s32(out_activation_min);
    const int32x4_t act_max = vdupq_n_s32(out_activation_max);

    while (count > 0)
    {
        const mve_pred16_t p = vctp32q((uint32_t)count);

        int32x4_t vect_1 = vldrbq_z_s32(input_1_vect, p);
        int32x4_t vect_2 = vldrbq_z_s32(input_2_vect, p);

        vect_1 = vshlq_r_s32(vaddq_n_s32(vect_1, input_1_offset), left_shift);
        vect_2 = vshlq_r_s32(vaddq_n_s32(vect_2, input_2_offset), left_shift);

        vect_1 = vqrdmulhq_n_s32(vect_1, input_1_mult);
        vect_2 = vqrdmulhq_n_s32(vect_2, input_2_mult);
        if (in_1_div)
        {
            vect_1 = arm_elementwise_add_s8_rdiv(vect_1, neg_exp_1);
        }
        if (in_2_div)
        {
            vect_2 = arm_elementwise_add_s8_rdiv(vect_2, neg_exp_2);
        }

        vect_1 = vaddq_s32(vect_1, vect_2);
        vect_1 = arm_elementwise_add_s8_rdiv(vqrdmulhq_n_s32(vect_1, out_mult), neg_exp_out);
        vect_1 = vaddq_n_s32(vect_1, out_offset);

        vect_1 = vmaxq_s32(vect_1, act_min);
        vect_1 = vminq_s32(vect_1, act_max);

        vstrbq_p_s32(output, vect_1, p);

        input_1_vect += 4;
        input_2_vect += 4;
        output += 4;
        count -= 4;
    }
}
#endif

arm_cmsis_nn_status arm_elementwise_add_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_1_mult,
                                           const int32_t input_1_shift,
                                           const int32_t input_2_offset,
                                           const int32_t input_2_mult,
                                           const int32_t input_2_shift,
                                           const int32_t left_shift,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    #if !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    if (input_1_shift <= 0 && input_1_shift >= -31 && input_2_shift <= 0 && input_2_shift >= -31 && left_shift >= 0 &&
        left_shift <= 31 && out_shift < 0 && out_shift >= -31)
    {
        #define ARM_ADD_S8_MVE_CALL(div_1, div_2)                                                                      \
            arm_elementwise_add_s8_mve(input_1_vect,                                                                   \
                                       input_2_vect,                                                                   \
                                       input_1_offset,                                                                 \
                                       input_1_mult,                                                                   \
                                       input_1_shift,                                                                  \
                                       input_2_offset,                                                                 \
                                       input_2_mult,                                                                   \
                                       input_2_shift,                                                                  \
                                       left_shift,                                                                     \
                                       output,                                                                         \
                                       out_offset,                                                                     \
                                       out_mult,                                                                       \
                                       out_shift,                                                                      \
                                       out_activation_min,                                                             \
                                       out_activation_max,                                                             \
                                       block_size,                                                                     \
                                       (div_1),                                                                        \
                                       (div_2))
        if (input_1_shift == 0 && input_2_shift == 0)
        {
            ARM_ADD_S8_MVE_CALL(0, 0);
        }
        else if (input_2_shift == 0)
        {
            ARM_ADD_S8_MVE_CALL(1, 0);
        }
        else if (input_1_shift == 0)
        {
            ARM_ADD_S8_MVE_CALL(0, 1);
        }
        else
        {
            ARM_ADD_S8_MVE_CALL(1, 1);
        }
        #undef ARM_ADD_S8_MVE_CALL
        return (ARM_CMSIS_NN_SUCCESS);
    }
    #endif
    int32_t count = block_size;

    while (count > 0)
    {
        int32x4_t vect_1;
        int32x4_t vect_2;

        mve_pred16_t p = vctp32q((uint32_t)count);

        vect_1 = vldrbq_z_s32(input_1_vect, p);
        vect_2 = vldrbq_z_s32(input_2_vect, p);

        vect_1 = vaddq_s32(vect_1, vdupq_n_s32(input_1_offset));
        vect_2 = vaddq_s32(vect_2, vdupq_n_s32(input_2_offset));

        vect_1 = vshlq_r_s32(vect_1, left_shift);
        vect_2 = vshlq_r_s32(vect_2, left_shift);

        vect_1 = arm_requantize_mve(vect_1, input_1_mult, input_1_shift);
        vect_2 = arm_requantize_mve(vect_2, input_2_mult, input_2_shift);

        vect_1 = vaddq_s32(vect_1, vect_2);
        vect_1 = arm_requantize_mve(vect_1, out_mult, out_shift);

        vect_1 = vaddq_n_s32(vect_1, out_offset);

        vect_1 = vmaxq_s32(vect_1, vdupq_n_s32(out_activation_min));
        vect_1 = vminq_s32(vect_1, vdupq_n_s32(out_activation_max));

        input_1_vect += 4;
        input_2_vect += 4;
        vstrbq_p_s32(output, vect_1, p);

        output += 4;
        count -= 4;
    }
#else
    int32_t loop_count;
    int32_t input_1;
    int32_t input_2;
    int32_t sum;

    #if defined(ARM_MATH_DSP)
    int32_t a_1, b_1, a_2, b_2;

    int32_t offset_1_packed, offset_2_packed;

    int8_t r1, r2, r3, r4;

    offset_1_packed = (int32_t)(((uint32_t)input_1_offset << 16) | ((uint32_t)input_1_offset & 0xFFFFu));
    offset_2_packed = (int32_t)(((uint32_t)input_2_offset << 16) | ((uint32_t)input_2_offset & 0xFFFFu));

    loop_count = block_size >> 2;

    while (loop_count > 0)
    {
        /* 4 outputs are calculated in one loop. The order of calculation is follows the order of output sign extension
           intrinsic */
        input_1_vect = read_and_pad_reordered(input_1_vect, &b_1, &a_1);
        input_2_vect = read_and_pad_reordered(input_2_vect, &b_2, &a_2);

        a_1 = SADD16(a_1, offset_1_packed);
        b_1 = SADD16(b_1, offset_1_packed);

        a_2 = SADD16(a_2, offset_2_packed);
        b_2 = SADD16(b_2, offset_2_packed);

        /* Sum 1 */
        input_1 = (int16_t)(b_1 & 0x0FFFF) * (int32_t)((uint32_t)1 << left_shift);

        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);

        input_2 = (int16_t)(b_2 & 0x0FFFF) * (int32_t)((uint32_t)1 << left_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum += out_offset;
        sum = ARM_NN_MAX(sum, out_activation_min);
        sum = ARM_NN_MIN(sum, out_activation_max);
        r1 = (int8_t)sum;

        /* Sum 3 */
        input_1 = (int16_t)(b_1 >> 16) * (int32_t)((uint32_t)1 << left_shift);
        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);

        input_2 = (int16_t)(b_2 >> 16) * (int32_t)((uint32_t)1 << left_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum += out_offset;
        sum = ARM_NN_MAX(sum, out_activation_min);
        sum = ARM_NN_MIN(sum, out_activation_max);
        r3 = (int8_t)sum;

        /* Sum 2 */
        input_1 = (int16_t)(a_1 & 0x0FFFF) * (int32_t)((uint32_t)1 << left_shift);
        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);

        input_2 = (int16_t)(a_2 & 0x0FFFF) * (int32_t)((uint32_t)1 << left_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum += out_offset;
        sum = ARM_NN_MAX(sum, out_activation_min);
        sum = ARM_NN_MIN(sum, out_activation_max);
        r2 = (int8_t)sum;

        /* Sum 4 */
        input_1 = (int16_t)(a_1 >> 16) * (int32_t)((uint32_t)1 << left_shift);
        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);

        input_2 = (int16_t)(a_2 >> 16) * (int32_t)((uint32_t)1 << left_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum += out_offset;
        sum = ARM_NN_MAX(sum, out_activation_min);
        sum = ARM_NN_MIN(sum, out_activation_max);
        r4 = (int8_t)sum;

        arm_nn_write_s8x4_ia(&output, PACK_S8x4_32x1(r1, r2, r3, r4));

        loop_count--;
    }

    loop_count = block_size & 0x3;
    #else
    loop_count = block_size;
    #endif

    while (loop_count > 0)
    {
        /* C = A + B */

        input_1 = (*input_1_vect++ + input_1_offset) * (int32_t)((uint32_t)1 << left_shift);
        input_2 = (*input_2_vect++ + input_2_offset) * (int32_t)((uint32_t)1 << left_shift);

        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum += out_offset;

        sum = ARM_NN_MAX(sum, out_activation_min);
        sum = ARM_NN_MIN(sum, out_activation_max);

        *output++ = (int8_t)sum;

        /* Decrement loop counter */
        loop_count--;
    }

#endif /* ARM_MATH_MVEI */

    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of Doxygen group
 */
