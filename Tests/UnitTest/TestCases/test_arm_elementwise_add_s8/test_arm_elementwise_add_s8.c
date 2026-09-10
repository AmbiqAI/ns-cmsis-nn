/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
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

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/add/test_data.h"
#include "../Utils/validate.h"

void add_arm_elementwise_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[ADD_DST_SIZE] = {0};

    const int8_t *input_data1 = add_input1;
    const int8_t *input_data2 = add_input2;

    const int32_t input_1_mult = ADD_INPUT1_MULT;
    const int32_t input_1_shift = ADD_INPUT1_SHIFT;
    const int32_t input_1_offset = ADD_INPUT1_OFFSET;
    const int32_t input_2_mult = ADD_INPUT2_MULT;
    const int32_t input_2_shift = ADD_INPUT2_SHIFT;
    const int32_t input_2_offset = ADD_INPUT2_OFFSET;

    const int32_t left_shift = ADD_LEFT_SHIFT;

    const int32_t out_offset = ADD_OUTPUT_OFFSET;
    const int32_t out_mult = ADD_OUTPUT_MULT;
    const int32_t out_shift = ADD_OUTPUT_SHIFT;

    const int32_t out_activation_min = ADD_OUT_ACTIVATION_MIN;
    const int32_t out_activation_max = ADD_OUT_ACTIVATION_MAX;

    arm_cmsis_nn_status result = arm_elementwise_add_s8(input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_1_mult,
                                                        input_1_shift,
                                                        input_2_offset,
                                                        input_2_mult,
                                                        input_2_shift,
                                                        left_shift,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        ADD_DST_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_output_ref, ADD_DST_SIZE));
}

/* Block-size invariance for the s8 add primitives.
 *
 * On Armv7E-M (ARM_MATH_DSP) these kernels process four elements at a time through a packed
 * halfword path and fall back to a scalar tail for the remainder, so a run of N elements takes a
 * different code path depending on N. The result must not depend on that: splitting a vector into
 * chunks of any size has to give the same answer as handing it over in one call. The reference
 * below is the one-element-per-call run, which only ever takes the scalar tail and is therefore
 * identical on every leg.
 *
 * The data deliberately drives "value + offset" negative for several elements inside a run of four
 * or more; that is the case the packed path used to get wrong. Small left_shift values matter as
 * much as the data: the packed path used to read a negative halfword as that value plus 2^16, and
 * once it is shifted left by 16 or more the excess leaves the int32 entirely, so a run at the
 * left_shift of 20 that TFLite Micro emits for int8 came out right anyway. Refs #343.
 */
#define BSI_LEN 20

typedef arm_cmsis_nn_status (*bsi_kernel_t)(const int8_t *input_1_vect,
                                            const int8_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_1_mult,
                                            const int32_t input_1_shift,
                                            const int32_t input_2_offset,
                                            const int32_t input_2_mult,
                                            const int32_t input_2_shift,
                                            const int32_t left_shift,
                                            int8_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

static const int8_t bsi_input_1[BSI_LEN] = {-100, -1,  5,  60, -128, 127,  0, -55, 33,  -7,
                                            12,   -90, 64, -3, 110,  -120, 8, 41,  -66, 2};
static const int8_t bsi_input_2[BSI_LEN] = {-7,  3,  -80,  20, 100, -128, 17,  4, -29, 71,
                                            -14, 55, -101, 9,  -6,  88,   -37, 0, 25,  -49};

static const int32_t bsi_input_1_offset = -3;
static const int32_t bsi_input_2_offset = 5;

/* Runs the whole vector through the kernel in chunks of `chunk` elements. The scalar-vs-vector
 * primitives take a single element as their first operand, which every chunk reuses.
 */
static void bsi_run(bsi_kernel_t kernel, bool input_1_is_scalar, int32_t left_shift, int32_t chunk, int8_t *output)
{
    for (int32_t i = 0; i < BSI_LEN; i += chunk)
    {
        const int32_t remaining = BSI_LEN - i;
        const int32_t block_size = chunk < remaining ? chunk : remaining;

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          kernel(input_1_is_scalar ? bsi_input_1 : bsi_input_1 + i,
                                 bsi_input_2 + i,
                                 bsi_input_1_offset,
                                 1 << 30,
                                 -1,
                                 bsi_input_2_offset,
                                 1 << 30,
                                 -1,
                                 left_shift,
                                 output + i,
                                 2,
                                 1 << 30,
                                 -1,
                                 -128,
                                 127,
                                 block_size));
    }
}

static void bsi_check(bsi_kernel_t kernel, bool input_1_is_scalar)
{
    /* left_shift 20 is what TFLite Micro emits for int8 add; the smaller values are in range for
     * the API and are where a lost sign bit actually changes the output.
     */
    static const int32_t left_shifts[] = {0, 1, 8, 15, 20};
    static const int32_t chunks[] = {2, 3, 4, 5, 7, 8, BSI_LEN};

    for (unsigned s = 0; s < sizeof(left_shifts) / sizeof(left_shifts[0]); s++)
    {
        int8_t reference[BSI_LEN] = {0};

        bsi_run(kernel, input_1_is_scalar, left_shifts[s], 1, reference);

        for (unsigned c = 0; c < sizeof(chunks) / sizeof(chunks[0]); c++)
        {
            int8_t actual[BSI_LEN] = {0};

            bsi_run(kernel, input_1_is_scalar, left_shifts[s], chunks[c], actual);
            TEST_ASSERT_TRUE(validate(actual, reference, BSI_LEN));
        }
    }
}

void block_size_invariance_arm_elementwise_add_s8(void) { bsi_check(arm_elementwise_add_s8, false); }

void block_size_invariance_arm_add_scalar_s8(void) { bsi_check(arm_add_scalar_s8, true); }
