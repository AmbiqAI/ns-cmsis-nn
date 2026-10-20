/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_reduce_bool.h"
#include "arm_nnfunctions.h"

arm_cmsis_nn_status arm_reduce_any_bool(const bool *input,
                                        const cmsis_nn_dims *input_dims,
                                        const cmsis_nn_dims *axis_dims,
                                        bool *output,
                                        const cmsis_nn_dims *output_dims)
{
    int32_t input_count, output_count;
    if (!input || !output ||
        !arm_nn_reduce_bool_dims_valid(input_dims, axis_dims, output_dims, &input_count, &output_count))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_nn_reduce_bool(input, input_dims, axis_dims, output, output_dims, input_count, output_count, false);
    return ARM_CMSIS_NN_SUCCESS;
}
