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
 * Title:        arm_elementwise_prelu_s8
 * Description:  Elementwise prelu
 *
 * $Date:        12 November 2025
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
 * s8 elementwise prelu for two vectors w/ same dimensions
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_elementwise_prelu_s8(const int8_t *input,
                                           const int8_t *alpha,
                                           const int32_t input_offset,
                                           const int32_t alpha_offset,
                                           const int32_t out_offset,
                                           const int32_t output_multiplier_identity,
                                           const int output_shift_identity,
                                           const int32_t output_multiplier_alpha,
                                           const int output_shift_alpha,
                                           int8_t * output,
                                           const int32_t block_size)
{
    for (int i = 0; i < block_size; ++i)
    {
        const int32_t input_value = input_offset + input[i];
        int32_t output_value;
        if (input_value >= 0) {
            output_value = arm_nn_requantize(input_value, output_multiplier_identity, output_shift_identity);
        }
        else {
            const int32_t alpha_value = alpha_offset + alpha[i];
            output_value = arm_nn_requantize(input_value * alpha_value, output_multiplier_alpha, output_shift_alpha);
        }
        output_value += out_offset;
        const int32_t clamped_output = MIN(INT8_MAX, MAX(INT8_MIN, output_value));
        output[i] = (int8_t)clamped_output;
    }

    return (ARM_CMSIS_NN_SUCCESS);
}
/**
 * @} end of Doxygen group
 */
