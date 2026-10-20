/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_logical_common.h"
#include "arm_nnfunctions.h"

static void logical_full(const bool *a, const bool *b, bool *out, int32_t count)
{
    arm_nn_logical_binary(a, b, out, count, false);
}

static void logical_scalar(const bool *scalar, const bool *vec, bool *out, int32_t count)
{
    arm_nn_logical_scalar(scalar, vec, out, count, false);
}

arm_cmsis_nn_status arm_logical_and_bool(const bool *input_1,
                                         const cmsis_nn_dims *input_1_dims,
                                         const bool *input_2,
                                         const cmsis_nn_dims *input_2_dims,
                                         bool *output,
                                         const cmsis_nn_dims *output_dims)
{
    if (!input_1 || !input_2 || !output || !arm_nn_logical_dims_valid(input_1_dims, input_2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    ARM_NN_BROADCAST_WALK_NHWC(bool,
                               bool,
                               input_1,
                               input_1_dims,
                               input_2,
                               input_2_dims,
                               output,
                               output_dims,
                               logical_full,
                               logical_scalar,
                               logical_scalar);
    return ARM_CMSIS_NN_SUCCESS;
}
