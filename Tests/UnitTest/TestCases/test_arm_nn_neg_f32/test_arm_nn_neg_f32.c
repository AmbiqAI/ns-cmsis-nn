/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#include "unity.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

static void neg_bits_f32(void)
{
    static const uint32_t patterns[] = {0,
                                        0x80000000u,
                                        1,
                                        0x80000001u,
                                        0x007fffffu,
                                        0x00800000u,
                                        0x3f800000u,
                                        0xbf800000u,
                                        0x7f7fffffu,
                                        0xff7fffffu,
                                        0x7f800000u,
                                        0xff800000u,
                                        0x7fc12345u,
                                        0xffc54321u,
                                        0x7f800001u,
                                        0xff800001u,
                                        0x12345678u};
    float32_t input[35], output[35];
    uint32_t raw[35], actual[35];
    for (uint32_t base = 0; base < 17; base += 1)
    {
        for (int32_t count = 1; count <= 33; ++count)
        {
            for (int32_t i = 0; i < 35; ++i)
            {
                raw[i] = patterns[(base + (uint32_t)i) % 17];
            }
            memcpy(input, raw, sizeof(raw));
            memset(output, 0xa5, sizeof(output));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_neg_f32(input + 1, output + 1, count));
            memcpy(actual, output, sizeof(actual));
            TEST_ASSERT_EQUAL_HEX32(0xa5a5a5a5u, actual[0]);
            TEST_ASSERT_EQUAL_HEX32(0xa5a5a5a5u, actual[count + 1]);
            for (int32_t i = 1; i <= count; ++i)
            {
                TEST_ASSERT_EQUAL_HEX32(raw[i] ^ 0x80000000u, actual[i]);
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_neg_f32(output + 1, output + 1, count));
            TEST_ASSERT_EQUAL_MEMORY(input + 1, output + 1, (size_t)count * sizeof(float32_t));
        }
    }
}

static void neg_args_f32(void)
{
    float32_t input[1], output[1];
    memset(input, 0x55, sizeof(input));
    memset(output, 0xa5, sizeof(output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_neg_f32(NULL, NULL, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f32(input, output, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f32(input, output, INT32_MAX));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f32(NULL, output, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_neg_f32(input, NULL, 1));
    uint32_t raw;
    memcpy(&raw, output, sizeof(raw));
    TEST_ASSERT_EQUAL_HEX32(0xa5a5a5a5u, raw);
}
