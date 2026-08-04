/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "split_f16_data.h"

void split_f16_arm_split_f16(void)
{
    float16_t out0[SPLIT_F16_SPLIT_SIZE] = {0};
    float16_t out1[SPLIT_F16_SPLIT_SIZE] = {0};
    float16_t out2[SPLIT_F16_SPLIT_SIZE] = {0};
    float16_t *outputs[SPLIT_F16_NUM_SPLITS] = {out0, out1, out2};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_split_f16(split_f16_input,
                                    SPLIT_F16_INPUT_DIMS,
                                    split_f16_input_shape,
                                    SPLIT_F16_AXIS,
                                    SPLIT_F16_NUM_SPLITS,
                                    split_f16_split_dims,
                                    outputs));

    for (int i = 0; i < SPLIT_F16_SPLIT_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)split_f16_output_ref_0[i], (float)out0[i]);
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)split_f16_output_ref_1[i], (float)out1[i]);
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)split_f16_output_ref_2[i], (float)out2[i]);
    }
}
