/*
 * Copyright (C) 2026 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include <stdio.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "unity.h"

#include "../TestData/sqrt_small_tensor_s16/test_data.h"
#include "../TestData/sqrt_long_row_s16/test_data.h"
#include "../TestData/sqrt_multi_batch_s16/test_data.h"
#include "../TestData/sqrt_tail_odd_s16/test_data.h"
#include "../TestData/sqrt_tail_mod7_s16/test_data.h"
#include "../TestData/sqrt_tail_mod5_s16/test_data.h"

#include "../Utils/validate.h"

void sqrt_small_tensor_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_SMALL_TENSOR_S16_IN_DIM;
    const int16_t *input_data = sqrt_small_tensor_s16_input_tensor;
    int16_t output_data[SQRT_SMALL_TENSOR_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result =
        arm_sqrt_s16(input_data, &input_dims, output_data, sqrt_small_tensor_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_small_tensor_s16_output, SQRT_SMALL_TENSOR_S16_OUTPUT_LEN));
}

void sqrt_long_row_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_LONG_ROW_S16_IN_DIM;
    const int16_t *input_data = sqrt_long_row_s16_input_tensor;
    int16_t output_data[SQRT_LONG_ROW_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result =
        arm_sqrt_s16(input_data, &input_dims, output_data, sqrt_long_row_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_long_row_s16_output, SQRT_LONG_ROW_S16_OUTPUT_LEN));
}

void sqrt_multi_batch_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_MULTI_BATCH_S16_IN_DIM;
    const int16_t *input_data = sqrt_multi_batch_s16_input_tensor;
    int16_t output_data[SQRT_MULTI_BATCH_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result =
        arm_sqrt_s16(input_data, &input_dims, output_data, sqrt_multi_batch_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_multi_batch_s16_output, SQRT_MULTI_BATCH_S16_OUTPUT_LEN));
}

// Regression for the GCC implicit tail-predication (dlstp.16/letp) miscompile:
// sizes with block_size % 8 in {3, 7} and inputs in the LUT-knee regime where
// slope * offset >= 2^16 (e.g. input 127 -> 653). The original vctp16q loop
// produced corrupted final elements here at -O2/-Ofast with GCC 14.2/15.2.
void sqrt_tail_odd_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_TAIL_ODD_S16_IN_DIM;
    const int16_t *input_data = sqrt_tail_odd_s16_input_tensor;
    int16_t output_data[SQRT_TAIL_ODD_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data, &input_dims, output_data, sqrt_tail_odd_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_tail_odd_s16_output, SQRT_TAIL_ODD_S16_OUTPUT_LEN));
}

// Regression for the GCC implicit tail-predication (dlstp.16/letp) miscompile:
// sizes with block_size % 8 in {3, 7} and inputs in the LUT-knee regime where
// slope * offset >= 2^16 (e.g. input 127 -> 653). The original vctp16q loop
// produced corrupted final elements here at -O2/-Ofast with GCC 14.2/15.2.
void sqrt_tail_mod7_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_TAIL_MOD7_S16_IN_DIM;
    const int16_t *input_data = sqrt_tail_mod7_s16_input_tensor;
    int16_t output_data[SQRT_TAIL_MOD7_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data, &input_dims, output_data, sqrt_tail_mod7_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_tail_mod7_s16_output, SQRT_TAIL_MOD7_S16_OUTPUT_LEN));
}

// Same regression family at residue 5 (13 % 8): every odd residue was
// corrupted by the dlstp miscompile, so one case each at residues 3, 5 and 7
// protects against a future width-special-cased tail.
void sqrt_tail_mod5_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_TAIL_MOD5_S16_IN_DIM;
    const int16_t *input_data = sqrt_tail_mod5_s16_input_tensor;
    int16_t output_data[SQRT_TAIL_MOD5_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data, &input_dims, output_data, sqrt_tail_mod5_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_tail_mod5_s16_output, SQRT_TAIL_MOD5_S16_OUTPUT_LEN));
}

// arm_sqrt_s16_tablefree: the MVE path must agree bit for bit with
// arm_nn_sqrt_s16_tablefree_element(), which is the plain C path, and every
// code <= 0 must give 0. Each case sweeps every int16 code, -32768..32767, in
// chunks of 4093 (511 full blocks plus a 5-lane predicated tail per call) and
// also folds the 32768 non-negative outputs into an FNV-1a hash (byte by byte,
// low byte first, offset basis 2166136261), so a target whose two paths agree
// with each other but not with a correct host is still caught. The hashes are
// the C path's outputs for codes 0..32767 computed on a host with a correctly
// rounded fmaf; a retune of the constants in arm_nnsupportfunctions.h must
// regenerate them the same way. scale = input_scale / output_scale^2.
typedef struct
{
    const char *name;
    float scale;
    uint32_t fnv1a;
} sqrt_tablefree_case;

static const sqrt_tablefree_case sqrt_tablefree_cases[] = {
    {"e2e op_sqrt_tensor_s16 (A=59.76)", 3571.42847f, 0x2A337D98u},
    {"e2e saturating from code 7060 (A=390)", 152100.016f, 0x01E95DBEu},
    {"unit saturating at 32763 (A=181.03)", 32771.8633f, 0x81ACD5E0u},
    {"unit A=3.0", 8.99999905f, 0x80FDC352u},
    {"unit A=1000", 1000000.06f, 0x7A371553u},
    {"grid A=18101.66, saturates from code 4", 327670000.0f, 0xB9733B92u},
};

static uint32_t sqrt_tablefree_fnv1a(uint32_t hash, const int16_t *values, int32_t count)
{
    for (int32_t i = 0; i < count; ++i)
    {
        const uint16_t u = (uint16_t)values[i];
        hash = (hash ^ (u & 0xFFu)) * 16777619u;
        hash = (hash ^ (u >> 8)) * 16777619u;
    }
    return hash;
}

void sqrt_tablefree_s16_every_code_arm_sqrt_s16_tablefree(void)
{
    enum
    {
        CHUNK = 4093
    };
    static int16_t input_data[CHUNK];
    static int16_t output_data[CHUNK];

    for (uint32_t c = 0; c < sizeof(sqrt_tablefree_cases) / sizeof(sqrt_tablefree_cases[0]); ++c)
    {
        const sqrt_tablefree_case *tc = &sqrt_tablefree_cases[c];
        uint32_t hash = 2166136261u;
        int32_t mismatches = 0;
        int32_t first_code = 0;
        int16_t first_got = 0;
        int16_t first_want = 0;

        for (int32_t start = -32768; start < 32768; start += CHUNK)
        {
            const int32_t len = (32768 - start) < CHUNK ? (32768 - start) : CHUNK;
            const cmsis_nn_dims input_dims = {1, 1, 1, len};

            for (int32_t i = 0; i < len; ++i)
            {
                input_data[i] = (int16_t)(start + i);
                output_data[i] = (int16_t)0x5555;
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              arm_sqrt_s16_tablefree(input_data, &input_dims, output_data, tc->scale));
            for (int32_t i = 0; i < len; ++i)
            {
                const int16_t want =
                    (input_data[i] <= 0) ? (int16_t)0 : arm_nn_sqrt_s16_tablefree_element(input_data[i], tc->scale);
                if (output_data[i] != want && mismatches++ == 0)
                {
                    first_code = start + i;
                    first_got = output_data[i];
                    first_want = want;
                }
            }
            if (start + len > 0)
            {
                const int32_t skip = start < 0 ? -start : 0;
                hash = sqrt_tablefree_fnv1a(hash, &output_data[skip], len - skip);
            }
        }
        if (mismatches)
        {
            printf("%s: %ld mismatches, first at code %ld: got %d want %d\n",
                   tc->name,
                   (long)mismatches,
                   (long)first_code,
                   first_got,
                   first_want);
        }
        TEST_ASSERT_EQUAL_INT32_MESSAGE(0, mismatches, tc->name);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(tc->fnv1a, hash, tc->name);
    }
}

// Every length from 0 to 17 and 24 (the sizes with every tail residue, a lone
// predicated block, and three full blocks) over a fixed pattern that mixes
// negative codes, 0, 1, the saturation knee and 32767. Output words past the
// length must stay untouched.
void sqrt_tablefree_s16_lengths_arm_sqrt_s16_tablefree(void)
{
    static const int16_t pattern[24] = {-32768, -1, 0, 1, 2,   3,    7059,  7060,  7061, 32767, 32766, 4,
                                        5,      6,  8, 9, 100, 1000, 10000, 16384, 7,    -7,    255,   256};
    static const int32_t lengths[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24};
    const float scale = 152100.016f; /* A = 390, saturates from code 7060 */
    int16_t output_data[32];

    for (uint32_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l)
    {
        const int32_t len = lengths[l];
        const cmsis_nn_dims input_dims = {1, 1, len, 1};

        for (int32_t i = 0; i < 32; ++i)
        {
            output_data[i] = (int16_t)0x5555;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_sqrt_s16_tablefree(pattern, &input_dims, output_data, scale));
        for (int32_t i = 0; i < len; ++i)
        {
            TEST_ASSERT_EQUAL_INT16(arm_nn_sqrt_s16_tablefree_element(pattern[i], scale), output_data[i]);
        }
        for (int32_t i = len; i < 32; ++i)
        {
            TEST_ASSERT_EQUAL_INT16((int16_t)0x5555, output_data[i]);
        }
    }
}

// A few codes against LiteRT's int16 SQRT (dequantize in float32, sqrtf, divide
// by the output scale, truncate, clamp; values taken from the interpreter):
// within 1 LSB is the contract, and the non-saturating e2e case and the
// saturating one both have to hold it.
void sqrt_tablefree_s16_litert_arm_sqrt_s16_tablefree(void)
{
    static const int16_t codes[22] = {0,   1,   2,   3,   4,    7,    8,    9,     15,    16,    100,
                                      127, 128, 255, 256, 1000, 4095, 4096, 10000, 16384, 32766, 32767};
    static const int16_t litert_e2e[22] = {0,   59,  84,  103, 119,  158,  169,  179,  231,  239,   597,
                                           673, 676, 954, 956, 1889, 3824, 3824, 5976, 7649, 10817, 10817};
    static const int16_t litert_sat[22] = {0,    390,  551,  675,  780,   1031,  1103,  1170,  1510,  1560,  3900,
                                           4395, 4412, 6227, 6240, 12332, 24956, 24960, 32767, 32767, 32767, 32767};
    const cmsis_nn_dims input_dims = {1, 1, 1, 22};
    int16_t output_data[22];

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_sqrt_s16_tablefree(codes, &input_dims, output_data, 3571.42847f));
    for (int32_t i = 0; i < 22; ++i)
    {
        TEST_ASSERT_INT16_WITHIN(1, litert_e2e[i], output_data[i]);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_sqrt_s16_tablefree(codes, &input_dims, output_data, 152100.016f));
    for (int32_t i = 0; i < 22; ++i)
    {
        TEST_ASSERT_INT16_WITHIN(1, litert_sat[i], output_data[i]);
    }
}
