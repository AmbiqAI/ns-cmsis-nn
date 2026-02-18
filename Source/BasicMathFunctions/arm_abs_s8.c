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
 * Title:        arm_abs_s8
 * Description:  Elementwise absolute value
 *
 * $Date:        2 February 2026
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
 * s8 elementwise absolute value
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_abs_s8(const int8_t *input,
                               const int32_t input_offset,
                               int8_t *output,
                               const int32_t out_offset,
                               const int32_t out_mult,
                               const int32_t out_shift,
                               const bool needs_rescale,
                               const int32_t out_activation_min,
                               const int32_t out_activation_max,
                               const int32_t block_size)
{
    int32_t loop_count = block_size;

    if (!needs_rescale)
    {
        while (loop_count > 0)
        {
            const int32_t input_val = (int32_t)*input++ - input_offset;
            int32_t output_val = (input_val > 0) ? input_val : -input_val;
            output_val += out_offset;
            output_val = MAX(output_val, out_activation_min);
            output_val = MIN(output_val, out_activation_max);

            *output++ = (int8_t)output_val;
            loop_count--;
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
    else
    {
        while (loop_count > 0)
        {
            const int32_t input_val = (int32_t)*input++ - input_offset;
            const int32_t abs_val = (input_val > 0) ? input_val : -input_val;
            int32_t output_val;
            output_val = arm_nn_requantize(abs_val, out_mult, out_shift);
            output_val += out_offset;
            output_val = MAX(output_val, out_activation_min);
            output_val = MIN(output_val, out_activation_max);

            *output++ = (int8_t)output_val;
            loop_count--;
        }

        return ARM_CMSIS_NN_SUCCESS;
    }
}

/**
 * @} end of Doxygen group
 */
