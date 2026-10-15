/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#include "unity.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

static void neg_bits_f16(void)
{

    float16_t input[35], output[35];
    uint16_t raw[35], actual[35];
    for (uint32_t base = 0; base < 65536; base += 31)
    {
        for (int32_t count = 1; count <= 33; ++count)
        {
            for (int32_t i = 0; i < 35; ++i)
            {
                raw[i] = (uint16_t)(base + (uint32_t)i);
            }
            memcpy(input, raw, sizeof(raw));
            memset(output, 0xa5, sizeof(output));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_neg_f16(input + 1, output + 1, count));
            memcpy(actual, output, sizeof(actual));
            TEST_ASSERT_EQUAL_HEX16(0xa5a5, actual[0]);
            TEST_ASSERT_EQUAL_HEX16(0xa5a5, actual[count + 1]);
            for (int32_t i = 1; i <= count; ++i)
            {
                TEST_ASSERT_EQUAL_HEX16(raw[i] ^ 0x8000u, actual[i]);
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_neg_f16(output + 1, output + 1, count));
            TEST_ASSERT_EQUAL_MEMORY(input + 1, output + 1, (size_t)count * sizeof(float16_t));
        }
    }
}

static void neg_args_f16(void)
{
    float16_t input[1], output[1];
    memset(input, 0x55, sizeof(input));
    memset(output, 0xa5, sizeof(output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_neg_f16(NULL, NULL, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f16(input, output, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f16(input, output, INT32_MAX));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f16(NULL, output, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f16(input, NULL, 1));
    uint16_t raw;
    memcpy(&raw, output, sizeof(raw));
    TEST_ASSERT_EQUAL_HEX16(0xa5a5, raw);
}
