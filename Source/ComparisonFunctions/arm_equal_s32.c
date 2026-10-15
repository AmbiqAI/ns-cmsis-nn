/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "Internal/arm_nn_compare_s32.h"
#include "arm_nnfunctions.h"

static void compare_full(const int32_t *a, const int32_t *b, bool *out, int32_t count)
{
    arm_nn_compare_s32(a, b, out, count, ARM_COMPARE_EQUAL, false, false);
}
static void compare_scalar_a(const int32_t *a, const int32_t *b, bool *out, int32_t count)
{
    arm_nn_compare_s32(a, b, out, count, ARM_COMPARE_EQUAL, true, false);
}
static void compare_scalar_b(const int32_t *b, const int32_t *a, bool *out, int32_t count)
{
    arm_nn_compare_s32(a, b, out, count, ARM_COMPARE_EQUAL, false, true);
}

arm_cmsis_nn_status arm_equal_s32(const int32_t *input_1,
                                  const cmsis_nn_dims *input_1_dims,
                                  const int32_t *input_2,
                                  const cmsis_nn_dims *input_2_dims,
                                  bool *output,
                                  const cmsis_nn_dims *output_dims)
{
    if (!input_1 || !input_2 || !output || !arm_nn_compare_s32_dims_valid(input_1_dims, input_2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    ARM_NN_BROADCAST_WALK_NHWC(int32_t,
                               bool,
                               input_1,
                               input_1_dims,
                               input_2,
                               input_2_dims,
                               output,
                               output_dims,
                               compare_full,
                               compare_scalar_a,
                               compare_scalar_b);
    return ARM_CMSIS_NN_SUCCESS;
}
