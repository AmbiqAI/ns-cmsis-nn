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
 * Title:        arm_prelu_scalar_s16
 * Description:  Elementwise prelu with scalar input
 *
 * $Date:        23 June 2026
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
 * s16 elementwise prelu with scalar
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_prelu_scalar_s16(const int16_t *scalar_vect,
                                         const int16_t *non_scalar_vect,
                                         const bool scalar_is_input,
                                         const int32_t input_offset,
                                         const int32_t alpha_offset,
                                         const int32_t output_offset,
                                         const int32_t output_multiplier_identity,
                                         const int32_t output_shift_identity,
                                         const int32_t output_multiplier_alpha,
                                         const int32_t output_shift_alpha,
                                         int16_t *output,
                                         const int32_t block_size)
{
    if (scalar_is_input)
    {
        const int32_t input_value = (int32_t)scalar_vect[0] + input_offset;

        if (input_value >= 0)
        {
            const int32_t output_value =
                arm_nn_requantize(input_value, output_multiplier_identity, output_shift_identity) + output_offset;

            for (int32_t i = 0; i < block_size; ++i)
            {
                int32_t acc = output_value;
                acc = MAX(acc, INT16_MIN);
                acc = MIN(acc, INT16_MAX);
                output[i] = (int16_t)acc;
            }
        }
        else
        {
            for (int32_t i = 0; i < block_size; ++i)
            {
                const int32_t alpha_value = (int32_t)non_scalar_vect[i] + alpha_offset;
                const int32_t prod = alpha_value * input_value;
                int32_t acc = arm_nn_requantize(prod, output_multiplier_alpha, output_shift_alpha) + output_offset;
                acc = MAX(acc, INT16_MIN);
                acc = MIN(acc, INT16_MAX);
                output[i] = (int16_t)acc;
            }
        }
    }
    else
    {
        const int32_t alpha_value = (int32_t)scalar_vect[0] + alpha_offset;

        for (int32_t i = 0; i < block_size; ++i)
        {
            const int32_t input_value = (int32_t)non_scalar_vect[i] + input_offset;

            int32_t acc;
            if (input_value >= 0)
            {
                acc = arm_nn_requantize(input_value, output_multiplier_identity, output_shift_identity) + output_offset;
            }
            else
            {
                const int32_t prod = alpha_value * input_value;
                acc = arm_nn_requantize(prod, output_multiplier_alpha, output_shift_alpha) + output_offset;
            }

            acc = MAX(acc, INT16_MIN);
            acc = MIN(acc, INT16_MAX);
            output[i] = (int16_t)acc;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */