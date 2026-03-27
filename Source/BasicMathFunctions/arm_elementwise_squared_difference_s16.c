/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_elementwise_squared_difference_s16
 * Description:  Elementwise squared difference
 *
 * $Date:        12 March 2026
 * $Revision:    V.1.0.0
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
 * s16 elementwise squared difference of two vectors.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_elementwise_squared_difference_s16(const int16_t *input_1_vect,
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
#if defined(ARM_MATH_MVEI)
    int32_t loop_count = block_size;

    while (loop_count > 0)
    {
        const mve_pred16_t p = vctp32q((uint32_t)loop_count);

        int32x4_t input_1 = vldrhq_z_s32(input_1_vect, p);
        int32x4_t input_2 = vldrhq_z_s32(input_2_vect, p);

        input_1 = vaddq_n_s32(input_1, input_1_offset);
        input_2 = vaddq_n_s32(input_2, input_2_offset);

        input_1 = vshlq_r_s32(input_1, left_shift);
        input_2 = vshlq_r_s32(input_2, left_shift);

        input_1 = arm_requantize_mve(input_1, input_1_mult, input_1_shift);
        input_2 = arm_requantize_mve(input_2, input_2_mult, input_2_shift);

        int32x4_t raw_diff = vsubq_s32(input_1, input_2);
        raw_diff = vmulq_s32(raw_diff, raw_diff);

        raw_diff = arm_requantize_mve(raw_diff, out_mult, out_shift);
        raw_diff = vaddq_n_s32(raw_diff, out_offset);
        raw_diff = vmaxq_s32(raw_diff, vdupq_n_s32(out_activation_min));
        raw_diff = vminq_s32(raw_diff, vdupq_n_s32(out_activation_max));

        vstrhq_p_s32(output, raw_diff, p);

        input_1_vect += 4;
        input_2_vect += 4;
        output += 4;
        loop_count -= 4;
    }
#else
    int32_t loop_count = block_size;

    while (loop_count > 0)
    {
        int32_t input_1 = (*input_1_vect++ + input_1_offset) << left_shift;
        int32_t input_2 = (*input_2_vect++ + input_2_offset) << left_shift;

        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        const int32_t raw_diff = input_1 - input_2;
        // Max of this is 32767^2 * (1 << 0), so won't overflow 32 bits.
        const int32_t squared_raw_diff = raw_diff * raw_diff;
        int32_t output_value = arm_nn_requantize(squared_raw_diff, out_mult, out_shift);
        output_value += out_offset;

        output_value = MAX(output_value, out_activation_min);
        output_value = MIN(output_value, out_activation_max);

        *output++ = (int16_t)output_value;
        loop_count--;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
