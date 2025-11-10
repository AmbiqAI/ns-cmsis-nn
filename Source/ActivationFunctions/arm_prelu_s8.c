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
 * Title:        arm_prelu_s8.c
 * Description:  Parametric ReLU function for int8_t data type
 *
 * $Date:        21 February 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */


/*
 * PReLU activation function for int8_t data type.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_prelu_s8(const cmsis_nn_dims *input_dims,
                                 const int8_t *input,
                                 const cmsis_nn_dims *alpha_dims,
                                 const int8_t *alpha,
                                 const int32_t input_offset,
                                 const int32_t alpha_offset,
                                 const int32_t output_offset,
                                 const int32_t output_multiplier_1,
                                 const int output_shift_1,
                                 const int32_t output_multiplier_2,
                                 const int output_shift_2,
                                 const cmsis_nn_dims *output_dims,
                                 int8_t *output)
{
    // PReLU is elementwise (input == output)
    if (input_dims->n != output_dims->n || input_dims->h != output_dims->h ||
        input_dims->w != output_dims->w || input_dims->c != output_dims->c) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Per-channel alpha ONLY
    if (!(alpha_dims->n == 1 && alpha_dims->h == 1 && alpha_dims->w == 1 &&
        alpha_dims->c == input_dims->c)) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int n = 0; n < output_dims->n; ++n) {
        for (int h = 0; h < output_dims->h; ++h) {
            for (int w = 0; w < output_dims->w; ++w) {
                for (int c = 0; c < output_dims->c; ++c) {
                    int output_index = c + (output_dims->c * w) + (output_dims->c * output_dims->w * h) +
                                   (output_dims->c * output_dims->w * output_dims->h * n);
                    int input_index = c + (input_dims->c * w) + (input_dims->c * input_dims->w * h) +
                                  (input_dims->c * input_dims->w * input_dims->h * n);
                    const int32_t input_value = input_offset + input[input_index];
                    int32_t output_value;
                    if (input_value >= 0) {
                        output_value = arm_nn_requantize(input_value, output_multiplier_1, output_shift_1);
                    }
                    else {
                        const int an = 0, ah = 0, aw = 0, ac = 0;
                        int alpha_index = ac + (alpha_dims->c * aw) + (alpha_dims->c * alpha_dims->w * ah)
                            + (alpha_dims->c * alpha_dims->w * alpha_dims->h * an);
                        const int32_t alpha_value = alpha_offset + alpha[alpha_index];
                        output_value = arm_nn_requantize(input_value * alpha_value, output_multiplier_2, output_shift_2);
                    }
                    output_value += output_offset;
                    const int32_t qmin = INT8_MIN;
                    const int32_t qmax = INT8_MAX;
                    int32_t clamped_output = CLAMP(output_value, qmax, qmin);
                    output[output_index] = clamped_output;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
