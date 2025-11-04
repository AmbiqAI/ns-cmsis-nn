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
 * Title:        arm_sub_scalar_s16
 * Description:  Elementwise subtraction with scalar input
 *
 * $Date:        4 November 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <stdbool.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

static arm_cmsis_nn_status arm_sub_scalar_s16_core(const int16_t *scalar_vect,
                                                   const int16_t *vector_vect,
                                                   const int32_t scalar_offset,
                                                   const int32_t scalar_mult,
                                                   const int32_t scalar_shift,
                                                   const int32_t vector_offset,
                                                   const int32_t vector_mult,
                                                   const int32_t vector_shift,
                                                   const int32_t left_shift,
                                                   int16_t *output,
                                                   const int32_t out_offset,
                                                   const int32_t out_mult,
                                                   const int32_t out_shift,
                                                   const int32_t out_activation_min,
                                                   const int32_t out_activation_max,
                                                   const int32_t block_size,
                                                   const bool scalar_minus_vector)
{
    (void)scalar_offset;
    (void)vector_offset;
    (void)out_offset;

    int32_t scalar_val = ((int32_t)scalar_vect[0]) << left_shift;
    scalar_val = arm_nn_requantize(scalar_val, scalar_mult, scalar_shift);

#if defined(ARM_MATH_MVEI)
    int32_t count = block_size;
    const int32x4_t scalar_vec = vdupq_n_s32(scalar_val);

    while (count > 0)
    {
        int32x4_t vect;
        int32x4_t rst;

        mve_pred16_t pred = vctp32q(count);

        vect = vldrhq_z_s32(vector_vect, pred);
        vect = vshlq_r_s32(vect, left_shift);
        vect = arm_requantize_mve(vect, vector_mult, vector_shift);

        rst = scalar_minus_vector ? vsubq_s32(scalar_vec, vect) : vsubq_s32(vect, scalar_vec);
        rst = arm_requantize_mve(rst, out_mult, out_shift);

        rst = vmaxq_s32(rst, vdupq_n_s32(out_activation_min));
        rst = vminq_s32(rst, vdupq_n_s32(out_activation_max));

        vector_vect += 4;
        vstrhq_p_s32(output, rst, pred);

        output += 4;
        count -= 4;
    }

#else
    int32_t loop_count;
    int32_t input_2;
    int32_t diff;
    int32_t two_halfword_vec;
    int16_t diff_1, diff_2;

    loop_count = block_size / 2;
    while (loop_count > 0)
    {
        two_halfword_vec = arm_nn_read_q15x2_ia(&vector_vect);

        input_2 = (int16_t)(two_halfword_vec & 0xFFFF) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);
        diff = scalar_minus_vector ? (scalar_val - input_2) : (input_2 - scalar_val);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);
        diff_1 = (int16_t)diff;

        input_2 = (int16_t)(two_halfword_vec >> 16) << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);
        diff = scalar_minus_vector ? (scalar_val - input_2) : (input_2 - scalar_val);
        diff = arm_nn_requantize(diff, out_mult, out_shift);
        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);
        diff_2 = (int16_t)diff;

        arm_nn_write_q15x2_ia(&output, PACK_Q15x2_32x1(diff_1, diff_2));

        loop_count--;
    }

    loop_count = block_size & 0x1;
    while (loop_count > 0)
    {
        /* C = scalar +/- vector */
        input_2 = *vector_vect++ << left_shift;
        input_2 = arm_nn_requantize(input_2, vector_mult, vector_shift);

        diff = scalar_minus_vector ? (scalar_val - input_2) : (input_2 - scalar_val);
        diff = arm_nn_requantize(diff, out_mult, out_shift);

        diff = MAX(diff, out_activation_min);
        diff = MIN(diff, out_activation_max);

        *output++ = (int16_t)diff;

        /* Decrement loop counter */
        loop_count--;
    }
#endif /* ARM_MATH_MVEI */

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * s16 elementwise subtract with scalar (scalar - vector)
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_sub_scalar_s16(const int16_t *input_1_vect,
                                       const int16_t *input_2_vect,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift,
                                       int16_t *output,
                                       const int32_t out_offset,
                                       const int32_t out_mult,
                                       const int32_t out_shift,
                                       const int32_t out_activation_min,
                                       const int32_t out_activation_max,
                                       const int32_t block_size)
{
    return arm_sub_scalar_s16_core(input_1_vect,
                                   input_2_vect,
                                   input_1_offset,
                                   input_1_mult,
                                   input_1_shift,
                                   input_2_offset,
                                   input_2_mult,
                                   input_2_shift,
                                   left_shift,
                                   output,
                                   out_offset,
                                   out_mult,
                                   out_shift,
                                   out_activation_min,
                                   out_activation_max,
                                   block_size,
                                   true);
}

/*
 * Internal helper for vector - scalar subtraction
 */
arm_cmsis_nn_status arm_rsub_scalar_s16(const int16_t *input_1_vect,
                                        const int16_t *input_2_vect,
                                        const int32_t input_1_offset,
                                        const int32_t input_1_mult,
                                        const int32_t input_1_shift,
                                        const int32_t input_2_offset,
                                        const int32_t input_2_mult,
                                        const int32_t input_2_shift,
                                        const int32_t left_shift,
                                        int16_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t out_activation_min,
                                        const int32_t out_activation_max,
                                        const int32_t block_size)
{
    return arm_sub_scalar_s16_core(input_2_vect,
                                   input_1_vect,
                                   input_2_offset,
                                   input_2_mult,
                                   input_2_shift,
                                   input_1_offset,
                                   input_1_mult,
                                   input_1_shift,
                                   left_shift,
                                   output,
                                   out_offset,
                                   out_mult,
                                   out_shift,
                                   out_activation_min,
                                   out_activation_max,
                                   block_size,
                                   false);
}

/**
 * @} end of Doxygen group
 */
