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
 * Title:        arm_squared_difference_scalar_s16
 * Description:  Elementwise squared difference with scalar input
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
 * s16 elementwise squared difference with scalar input.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_squared_difference_scalar_s16(const int16_t *input_1_vect,
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
    const int32_t scalar_value =
        arm_nn_requantize(((int32_t)input_1_vect[0] + input_1_offset) << left_shift, input_1_mult, input_1_shift);
    int32_t loop_count = block_size;

    while (loop_count > 0)
    {
        int32_t input_2 = (*input_2_vect++ + input_2_offset) << left_shift;
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        const int32_t raw_diff = scalar_value - input_2;
        // Max of this is 32767^2 * (1 << 0), so won't overflow 32 bits.
        const int32_t squared_raw_diff = raw_diff * raw_diff;
        int32_t output_value = arm_nn_requantize(squared_raw_diff, out_mult, out_shift);
        output_value += out_offset;

        output_value = MAX(output_value, out_activation_min);
        output_value = MIN(output_value, out_activation_max);

        *output++ = (int16_t)output_value;
        loop_count--;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
