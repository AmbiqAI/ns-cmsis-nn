/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/transpose_3dim/test_data.h"
#include "../TestData/transpose_3dim2/test_data.h"
#include "../TestData/transpose_chwn/test_data.h"
#include "../TestData/transpose_default/test_data.h"
#include "../TestData/transpose_matrix/test_data.h"
#include "../TestData/transpose_nchw/test_data.h"
#include "../TestData/transpose_ncwh/test_data.h"
#include "../TestData/transpose_nhcw/test_data.h"
#include "../TestData/transpose_nwhc/test_data.h"
#include "../TestData/transpose_wchn/test_data.h"
#include "../Utils/validate.h"

void transpose_default_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_DEFAULT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_DEFAULT_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_DEFAULT_OUT_DIM;

    const int8_t *input_data = transpose_default_input_tensor;
    const int8_t *const output_ref = transpose_default_output;
    const int32_t output_ref_size = TRANSPOSE_DEFAULT_SIZE;

    const uint32_t perm[TRANSPOSE_DEFAULT_PERM_SIZE] = TRANSPOSE_DEFAULT_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_DEFAULT_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_nhcw_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NHCW_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NHCW_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NHCW_OUT_DIM;

    const int8_t *input_data = transpose_nhcw_input_tensor;
    const int8_t *const output_ref = transpose_nhcw_output;
    const int32_t output_ref_size = TRANSPOSE_NHCW_SIZE;

    const uint32_t perm[TRANSPOSE_NHCW_PERM_SIZE] = TRANSPOSE_NHCW_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NHCW_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_wchn_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_WCHN_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_WCHN_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_WCHN_OUT_DIM;

    const int8_t *input_data = transpose_wchn_input_tensor;
    const int8_t *const output_ref = transpose_wchn_output;
    const int32_t output_ref_size = TRANSPOSE_WCHN_SIZE;

    const uint32_t perm[TRANSPOSE_WCHN_PERM_SIZE] = TRANSPOSE_WCHN_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_WCHN_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_nchw_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NCHW_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NCHW_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NCHW_OUT_DIM;

    const int8_t *input_data = transpose_nchw_input_tensor;
    const int8_t *const output_ref = transpose_nchw_output;
    const int32_t output_ref_size = TRANSPOSE_NCHW_SIZE;

    const uint32_t perm[TRANSPOSE_NCHW_PERM_SIZE] = TRANSPOSE_NCHW_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NCHW_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_chwn_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_CHWN_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_CHWN_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_CHWN_OUT_DIM;

    const int8_t *input_data = transpose_chwn_input_tensor;
    const int8_t *const output_ref = transpose_chwn_output;
    const int32_t output_ref_size = TRANSPOSE_CHWN_SIZE;

    const uint32_t perm[TRANSPOSE_CHWN_PERM_SIZE] = TRANSPOSE_CHWN_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_CHWN_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_matrix_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_MATRIX_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_MATRIX_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_MATRIX_OUT_DIM;

    const int8_t *input_data = transpose_matrix_input_tensor;
    const int8_t *const output_ref = transpose_matrix_output;
    const int32_t output_ref_size = TRANSPOSE_MATRIX_SIZE;

    const uint32_t perm[TRANSPOSE_MATRIX_PERM_SIZE] = TRANSPOSE_MATRIX_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_MATRIX_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_ncwh_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NCWH_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NCWH_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NCWH_OUT_DIM;

    const int8_t *input_data = transpose_ncwh_input_tensor;
    const int8_t *const output_ref = transpose_ncwh_output;
    const int32_t output_ref_size = TRANSPOSE_NCWH_SIZE;

    const uint32_t perm[TRANSPOSE_NCWH_PERM_SIZE] = TRANSPOSE_NCWH_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NCWH_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_nwhc_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NWHC_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NWHC_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NWHC_OUT_DIM;

    const int8_t *input_data = transpose_nwhc_input_tensor;
    const int8_t *const output_ref = transpose_nwhc_output;
    const int32_t output_ref_size = TRANSPOSE_NWHC_SIZE;

    const uint32_t perm[TRANSPOSE_NWHC_PERM_SIZE] = TRANSPOSE_NWHC_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NWHC_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_3dim_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_3DIM_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_3DIM_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_3DIM_OUT_DIM;

    const int8_t *input_data = transpose_3dim_input_tensor;
    const int8_t *const output_ref = transpose_3dim_output;
    const int32_t output_ref_size = TRANSPOSE_3DIM_SIZE;

    const uint32_t perm[TRANSPOSE_3DIM_PERM_SIZE] = TRANSPOSE_3DIM_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_3DIM_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_3dim2_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_3DIM2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_3DIM2_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_3DIM2_OUT_DIM;

    const int8_t *input_data = transpose_3dim2_input_tensor;
    const int8_t *const output_ref = transpose_3dim2_output;
    const int32_t output_ref_size = TRANSPOSE_3DIM2_SIZE;

    const uint32_t perm[TRANSPOSE_3DIM2_PERM_SIZE] = TRANSPOSE_3DIM2_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_3DIM2_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

// An output_dims that is not input_dims permuted by perm must be rejected before any write.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_dims_mismatch_arm_transpose_s8(void)
{
    const int32_t buffer_size = 1 * 8 * 10 * 12;
    const int8_t poison = (int8_t)0x5a;
    static int8_t input_data[1 * 8 * 10 * 12];
    static int8_t output_data[1 * 8 * 10 * 12];

    const cmsis_nn_dims input_dims = {1, 8, 10, 12};
    const cmsis_nn_dims output_dims = {12, 8, 10, 1};
    const uint32_t perm[4] = {3, 2, 1, 0};
    const cmsis_nn_transpose_params transpose_params = {4, perm};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        input_data[i] = (int8_t)(i & 0x7f);
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}

// The 2-D path transposes unconditionally, so the identity permutation needs its own case.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_2dim_identity_arm_transpose_s8(void)
{
    const int8_t input_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t output_data[6] = {0};

    const cmsis_nn_dims input_dims = {2, 3, 1, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    const uint32_t perm[2] = {0, 1};
    const cmsis_nn_transpose_params transpose_params = {2, perm};

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_data, input_data, 6));
}

// A perm with a repeated axis is not a permutation; the dims agree, so only the repeat can reject it.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_perm_duplicate_axis_arm_transpose_s8(void)
{
    const int32_t input_size = 2 * 2 * 3 * 4;
    const int32_t output_size = 2 * 2 * 2 * 3;
    const int8_t poison = (int8_t)0x5a;
    static int8_t input_data[2 * 2 * 3 * 4];
    static int8_t output_data[2 * 2 * 2 * 3];

    const cmsis_nn_dims input_dims = {2, 2, 3, 4};
    const cmsis_nn_dims output_dims = {2, 2, 2, 3};
    const uint32_t perm[4] = {0, 0, 1, 2};
    const cmsis_nn_transpose_params transpose_params = {4, perm};

    for (int32_t i = 0; i < input_size; i++)
    {
        input_data[i] = (int8_t)(i & 0x7f);
    }
    for (int32_t i = 0; i < output_size; i++)
    {
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < output_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}

// Only num_dims in [1, 4] is representable in cmsis_nn_dims. see AmbiqAI/ns-cmsis-nn#443
void transpose_num_dims_out_of_range_arm_transpose_s8(void)
{
    const int32_t buffer_size = 2 * 3 * 4 * 5;
    const int8_t poison = (int8_t)0x5a;
    static int8_t input_data[2 * 3 * 4 * 5];
    static int8_t output_data[2 * 3 * 4 * 5];

    const cmsis_nn_dims input_dims = {2, 3, 4, 5};
    const cmsis_nn_dims output_dims = {2, 3, 4, 5};
    const uint32_t perm[4] = {0, 1, 2, 3};
    const cmsis_nn_transpose_params too_few = {0, perm};
    const cmsis_nn_transpose_params too_many = {5, perm};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        input_data[i] = (int8_t)(i & 0x7f);
        output_data[i] = poison;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &too_few));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &too_many));

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}
// An extent of 0 agrees across the permutation, so only the extent check can reject it; without it the copy length
// wraps. see AmbiqAI/ns-cmsis-nn#443
void transpose_zero_extent_arm_transpose_s8(void)
{
    const int32_t buffer_size = 6;
    const int8_t poison = (int8_t)0x5a;
    int8_t input_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t output_data[6];

    const cmsis_nn_dims input_dims = {2, 0, 1, 1};
    const cmsis_nn_dims output_dims = {0, 2, 1, 1};
    const uint32_t perm[2] = {1, 0};
    const cmsis_nn_transpose_params transpose_params = {2, perm};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}

/*
 * Dispatch coverage for AmbiqAI/ns-cmsis-nn#757. Expected outputs come from the index formula, not from the kernel's
 * routing, and every call checks a canary past the end of the output.
 */
#define TRANSPOSE_CANARY_SIZE (16)

#if defined(USING_FVP_CORSTONE_300)
    #define TRANSPOSE_WIDE_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define TRANSPOSE_WIDE_SECTION
#endif

/* Fits [1,1,8,9363], one column past the 16-bit gather offset limit, and [65537,1] and [1,1,2,65537], a row and a
 * column count past 16 bits. */
#define TRANSPOSE_WIDE_SIZE (2 * 65537)
static int8_t transpose_wide_input[TRANSPOSE_WIDE_SIZE] TRANSPOSE_WIDE_SECTION;
static int8_t transpose_wide_output[TRANSPOSE_WIDE_SIZE + TRANSPOSE_CANARY_SIZE] TRANSPOSE_WIDE_SECTION;
static int8_t transpose_wide_ref[TRANSPOSE_WIDE_SIZE] TRANSPOSE_WIDE_SECTION;

static void transpose_fill_s8(int8_t *data, int32_t size)
{
    for (int32_t i = 0; i < size; i++)
    {
        data[i] = (int8_t)(i * 37 + (i >> 8) * 11 + (i >> 16));
    }
}

/* out[o] = in[sum over i of index_i * input_stride[perm[i]]], walking the output in order. */
static void
transpose_ref_s8(const int8_t *input, int8_t *output, const int32_t *in_dims, const uint32_t *perm, int32_t num_dims)
{
    int32_t in_strides[4];
    int32_t out_dims[4];
    int32_t index[4] = {0, 0, 0, 0};
    int32_t size = 1;

    for (int32_t i = num_dims - 1; i >= 0; i--)
    {
        in_strides[i] = size;
        size *= in_dims[i];
    }
    for (int32_t i = 0; i < num_dims; i++)
    {
        out_dims[i] = in_dims[perm[i]];
    }

    for (int32_t o = 0; o < size; o++)
    {
        int32_t from = 0;
        for (int32_t i = 0; i < num_dims; i++)
        {
            from += index[i] * in_strides[perm[i]];
        }
        output[o] = input[from];

        for (int32_t i = num_dims - 1; i >= 0; i--)
        {
            if (++index[i] < out_dims[i])
            {
                break;
            }
            index[i] = 0;
        }
    }
}

static void transpose_check_s8(const int8_t *input,
                               int8_t *output,
                               int8_t *ref,
                               const int32_t *in_dims,
                               const uint32_t *perm,
                               int32_t num_dims)
{
    const int8_t poison = (int8_t)0x5a;
    int32_t dims[4] = {1, 1, 1, 1};
    int32_t out[4] = {1, 1, 1, 1};
    int32_t size = 1;
    char perm_text[5] = {0};
    char message[128];

    for (int32_t i = 0; i < num_dims; i++)
    {
        dims[i] = in_dims[i];
        out[i] = in_dims[perm[i]];
        size *= in_dims[i];
        perm_text[i] = (char)('0' + perm[i]);
    }
    const cmsis_nn_dims input_dims = {dims[0], dims[1], dims[2], dims[3]};
    const cmsis_nn_dims output_dims = {out[0], out[1], out[2], out[3]};
    const cmsis_nn_transpose_params transpose_params = {num_dims, perm};

    for (int32_t i = 0; i < size + TRANSPOSE_CANARY_SIZE; i++)
    {
        output[i] = poison;
    }
    transpose_ref_s8(input, ref, in_dims, perm, num_dims);

    snprintf(message,
             sizeof(message),
             "perm %s dims %ld,%ld,%ld,%ld",
             perm_text,
             (long)dims[0],
             (long)dims[1],
             (long)dims[2],
             (long)dims[3]);

    TEST_ASSERT_EQUAL_MESSAGE(
        ARM_CMSIS_NN_SUCCESS, arm_transpose_s8(input, output, &input_dims, &output_dims, &transpose_params), message);
    TEST_ASSERT_TRUE_MESSAGE(validate(output, ref, size), message);

    bool canary_intact = true;
    for (int32_t i = size; i < size + TRANSPOSE_CANARY_SIZE; i++)
    {
        canary_intact = canary_intact && (output[i] == poison);
    }
    TEST_ASSERT_TRUE_MESSAGE(canary_intact, message);
}

/* Every rank-3 and rank-4 permutation, with a row count that is not a multiple of the 8-lane gather and a batch. */
void transpose_all_permutations_arm_transpose_s8(void)
{
    static int8_t input[2 * 3 * 11 * 5];
    static int8_t output[2 * 3 * 11 * 5 + TRANSPOSE_CANARY_SIZE];
    static int8_t ref[2 * 3 * 11 * 5];
    const int32_t dims3[3] = {2, 11, 5};
    const int32_t dims4[4] = {2, 3, 11, 5};

    transpose_fill_s8(input, 2 * 3 * 11 * 5);

    for (uint32_t a = 0; a < 3; a++)
    {
        for (uint32_t b = 0; b < 3; b++)
        {
            if (b == a)
            {
                continue;
            }
            const uint32_t perm[3] = {a, b, 3 - a - b};
            transpose_check_s8(input, output, ref, dims3, perm, 3);
        }
    }

    for (uint32_t a = 0; a < 4; a++)
    {
        for (uint32_t b = 0; b < 4; b++)
        {
            for (uint32_t c = 0; c < 4; c++)
            {
                if (b == a || c == a || c == b)
                {
                    continue;
                }
                const uint32_t perm[4] = {a, b, c, 6 - a - b - c};
                transpose_check_s8(input, output, ref, dims4, perm, 4);
            }
        }
    }
}

/* The channel/history swaps of the WeKWS DS-TCN, [1,256,W] -> [1,W,256]. see AmbiqAI/helia-core#3 */
void transpose_history_arm_transpose_s8(void)
{
    static int8_t input[256 * 57];
    static int8_t output[256 * 57 + TRANSPOSE_CANARY_SIZE];
    static int8_t ref[256 * 57];
    const int32_t widths[4] = {8, 15, 29, 57};
    const uint32_t perm[3] = {0, 2, 1};

    transpose_fill_s8(input, 256 * 57);

    for (int32_t i = 0; i < 4; i++)
    {
        const int32_t dims[3] = {1, 256, widths[i]};
        transpose_check_s8(input, output, ref, dims, perm, 3);
    }
}

/* The gather path holds lane offsets in 16 bits and must hand wider shapes to the general loop. */
void transpose_wide_arm_transpose_s8(void)
{
    const uint32_t swap_last[4] = {0, 1, 3, 2};
    const uint32_t swap_2d[2] = {1, 0};
    const int32_t widest_gather[4] = {1, 1, 8, 9362};
    const int32_t past_gather[4] = {1, 1, 8, 9363};
    const int32_t long_rows[2] = {65537, 1};
    const int32_t long_cols[4] = {1, 1, 2, 65537};

    transpose_fill_s8(transpose_wide_input, TRANSPOSE_WIDE_SIZE);

    transpose_check_s8(transpose_wide_input, transpose_wide_output, transpose_wide_ref, widest_gather, swap_last, 4);
    transpose_check_s8(transpose_wide_input, transpose_wide_output, transpose_wide_ref, past_gather, swap_last, 4);
    transpose_check_s8(transpose_wide_input, transpose_wide_output, transpose_wide_ref, long_rows, swap_2d, 2);
    transpose_check_s8(transpose_wide_input, transpose_wide_output, transpose_wide_ref, long_cols, swap_last, 4);
}

#if defined(USING_FVP_CORSTONE_300) && defined(ARM_MATH_MVEI)
/* MVE loads retired during one call, from the core PMU. The FVP counts MVE loads but not the gather subset. */
static uint32_t transpose_mve_loads_s8(const int32_t *in_dims, const uint32_t *perm, int32_t num_dims)
{
    int32_t dims[4] = {1, 1, 1, 1};
    int32_t out[4] = {1, 1, 1, 1};
    for (int32_t i = 0; i < num_dims; i++)
    {
        dims[i] = in_dims[i];
        out[i] = in_dims[perm[i]];
    }
    const cmsis_nn_dims input_dims = {dims[0], dims[1], dims[2], dims[3]};
    const cmsis_nn_dims output_dims = {out[0], out[1], out[2], out[3]};
    const cmsis_nn_transpose_params transpose_params = {num_dims, perm};

    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    ARM_PMU_Set_EVTYPER(0, ARM_PMU_MVE_LD_RETIRED);
    ARM_PMU_EVCNTR_ALL_Reset();
    ARM_PMU_CNTR_Enable(PMU_CNTENSET_CNT0_ENABLE_Msk);
    ARM_PMU_Enable();

    const arm_cmsis_nn_status result =
        arm_transpose_s8(transpose_wide_input, transpose_wide_output, &input_dims, &output_dims, &transpose_params);

    ARM_PMU_CNTR_Disable(PMU_CNTENCLR_CNT0_ENABLE_Msk);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    return ARM_PMU_Get_EVCNTR(0);
}
#endif

/*
 * Numerics agree on either path, so the route is checked by MVE load count. Swapping the last two axes on the gather
 * path issues outer * cols * ceil(rows / 8) gathers, and a copy issues one load per 16 bytes. Other routes are left to
 * the numeric tests, since a compiler may vectorize the general loop.
 */
void transpose_route_arm_transpose_s8(void)
{
#if defined(USING_FVP_CORSTONE_300) && defined(ARM_MATH_MVEI)
    const uint32_t swap_last[4] = {0, 1, 3, 2};
    const uint32_t identity[4] = {0, 1, 2, 3};
    const uint32_t swap_last_3d[3] = {0, 2, 1};
    const int32_t small[4] = {2, 3, 11, 5};
    const int32_t small_3d[3] = {2, 11, 5};
    const int32_t history[3] = {1, 256, 57};
    const int32_t widest_gather[4] = {1, 1, 8, 9362};

    /* 2 * 3 * 5 * 2, 1 * 57 * 32, 2 * 5 * 2 and 1 * 9362 * 1 gathers. */
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(60, transpose_mve_loads_s8(small, swap_last, 4));
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(1824, transpose_mve_loads_s8(history, swap_last_3d, 3));
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(20, transpose_mve_loads_s8(small_3d, swap_last_3d, 3));
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(9362, transpose_mve_loads_s8(widest_gather, swap_last, 4));

    /* Identities copy 330 and 66 bytes, fewer loads than the 60 and 22 gathers a swap of those shapes takes. */
    const uint32_t rank4_copy = transpose_mve_loads_s8(small, identity, 4);
    const uint32_t rank3_copy = transpose_mve_loads_s8(small, identity, 3);
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(20, rank4_copy);
    TEST_ASSERT_LESS_THAN_UINT32(60, rank4_copy);
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(4, rank3_copy);
    TEST_ASSERT_LESS_THAN_UINT32(22, rank3_copy);
#else
    TEST_IGNORE_MESSAGE("The gather route exists only on MVE targets, and the PMU is read only on the FVP.");
#endif
}
