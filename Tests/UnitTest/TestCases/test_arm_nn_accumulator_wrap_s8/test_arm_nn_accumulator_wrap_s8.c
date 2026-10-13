/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq Micro, Inc.
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnsupportfunctions.h"
#include "unity.h"
#define REDUCTION_COLUMNS 65794

/* One arena for this dedicated suite, within Corstone's DTCM rather than its 64 KiB heap. */
static int8_t reduction_buffer[5 * REDUCTION_COLUMNS + 32];

static int32_t wrap_reference(const int64_t sum)
{
    const uint32_t bits = (uint32_t)sum;
    int32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Independent Q31 multiplication for these shift-zero cases, rounding ties upward. */
static int32_t quantized_reference(const int64_t sum, const int32_t multiplier, const int32_t low, const int32_t high)
{
    const int64_t numerator = (int64_t)wrap_reference(sum) * multiplier + (INT64_C(1) << 30);
    const int64_t denominator = INT64_C(1) << 31;
    int64_t value = numerator / denominator;
    if (numerator < 0 && numerator % denominator)
    {
        --value;
    }
    if (value < low)
    {
        value = low;
    }
    if (value > high)
    {
        value = high;
    }
    return (int32_t)value;
}

static void check_vector_wrap(const bool per_channel)
{
    const int32_t bias[5] = {INT32_MIN, INT32_MAX, 0, 8, INT32_MIN};
    const int32_t shifts[5] = {0};
    const int32_t per_mult[5] = {1073741824, 536870912, 1073741824, 536870912, 1073741824};
    for (int columns = 1; columns <= 6; columns += 5)
    {
        int8_t lhs[32] = {1};
        int8_t rhs[64] = {0};
        if (columns == 6)
        {
            lhs[0] = 0;
            lhs[4] = 1;
            lhs[5] = -1;
        }
        for (int row = 0; row < 5; ++row)
        {
            rhs[row * columns + columns - 1] = -1;
            if (columns == 6)
            {
                rhs[row * columns + 4] = -1;
            }
        }
        for (int rhs_offset = 0; rhs_offset <= 1; ++rhs_offset)
        {
            int32_t kernel_sum[5];
            memcpy(kernel_sum, bias, sizeof(bias)); /* lhs_offset is zero, so kernel_sum is bias. */
            int8_t output[10];
            memset(output, 77, sizeof(output));
            arm_cmsis_nn_status status;
            if (per_channel)
            {
                status = arm_nn_vec_mat_mult_t_per_ch_s8(
                    lhs, rhs, kernel_sum, bias, output, 0, 0, per_mult, shifts, columns, 5, -128, 127, 2, rhs_offset);
            }
            else
            {
                status = arm_nn_vec_mat_mult_t_s8(
                    lhs, rhs, kernel_sum, bias, output, 0, 0, 1073741824, 0, columns, 5, -128, 127, 2, rhs_offset);
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            for (int row = 0; row < 5; ++row)
            {
                int64_t sum = bias[row];
                for (int col = 0; col < columns; ++col)
                {
                    sum += (int64_t)lhs[col] * ((int32_t)rhs[row * columns + col] + rhs_offset);
                }
                TEST_ASSERT_EQUAL_INT8(quantized_reference(sum, per_channel ? per_mult[row] : 1073741824, -128, 127),
                                       output[2 * row]);
                TEST_ASSERT_EQUAL_INT8(77, output[2 * row + 1]);
            }
        }
    }
    const int long_columns = 131073;
    int8_t *long_lhs = reduction_buffer;
    int8_t *long_rhs = reduction_buffer + long_columns + 16;
    memset(long_lhs, -128, long_columns + 16);
    memset(long_rhs, 0, long_columns + 16);
    const int32_t zero[1] = {0};
    const int32_t multiplier[1] = {1073741824};
    int8_t long_output[2] = {77, 77};
    const arm_cmsis_nn_status status = per_channel
        ? arm_nn_vec_mat_mult_t_per_ch_s8(
              long_lhs, long_rhs, zero, zero, long_output, 0, 0, multiplier, zero, long_columns, 1, -128, 127, 2, 128)
        : arm_nn_vec_mat_mult_t_s8(
              long_lhs, long_rhs, zero, zero, long_output, 0, 0, multiplier[0], 0, long_columns, 1, -128, 127, 2, 128);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT8(quantized_reference(-INT64_C(16384) * long_columns, multiplier[0], -128, 127),
                           long_output[0]);
    TEST_ASSERT_EQUAL_INT8(77, long_output[1]);
}

void vec_mat_accumulator_wrap_s8(void) { check_vector_wrap(false); }
void per_ch_accumulator_wrap_s8(void) { check_vector_wrap(true); }

void svdf_accumulator_wrap_s8(void)
{
    const int columns = REDUCTION_COLUMNS;
    int8_t *lhs = reduction_buffer;
    int8_t *rhs = reduction_buffer + columns + 16;
    memset(lhs, -128, columns + 16);
    memset(rhs, -128, 3 * columns + 16);
    int16_t outputs[2][6] = {{0, 77, 0, 77, 0, 77}, {0, 77, 0, 77, 0, 77}};
    arm_cmsis_nn_status status[2];
    for (int group = 0; group < 2; ++group)
    {
        status[group] = arm_nn_vec_mat_mult_t_svdf_s8(
            lhs, rhs, outputs[group], -127, 2, 1073741824, 0, columns, 1 + 2 * group, -32768, 32767);
    }
    const int64_t sum = INT64_C(32640) * columns;
    for (int group = 0; group < 2; ++group)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status[group]);
        for (int row = 0; row < 1 + 2 * group; ++row)
        {
            TEST_ASSERT_EQUAL_INT16(quantized_reference(sum, 1073741824, -32768, 32767), outputs[group][2 * row]);
            TEST_ASSERT_EQUAL_INT16(77, outputs[group][2 * row + 1]);
        }
    }
}

void mat_mult_accumulator_wrap_s32(void)
{
    const int columns = REDUCTION_COLUMNS;
    int8_t *lhs = reduction_buffer;
    int8_t *rhs = reduction_buffer + 4 * columns + 16;
    memset(lhs, -128, 4 * columns + 16);
    memset(rhs, -128, columns + 16);
    const int32_t expected = wrap_reference(INT64_C(32640) * columns);
    for (int rows = 3; rows <= 4; ++rows)
    {
        int32_t output[12];
        for (int i = 0; i < 12; ++i)
        {
            output[i] = 77;
        }
        for (int row = 0; row < rows; ++row)
        {
            output[3 * row] = 0; /* Owning contract requires zero initial output. */
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_mat_mult_nt_t_s8_s32(lhs, rhs, output, rows, columns, 1, -127, 3));
        for (int i = 0; i < 12; ++i)
        {
            TEST_ASSERT_EQUAL_INT32((i < 3 * rows && i % 3 == 0) ? expected : 77, output[i]);
        }
    }
}

void lstm_accumulator_wrap_s8_s16(void)
{
    const int8_t lhs[32] = {1};
    const int8_t rhs[32] = {-1, -1, -1, -1, -1};
    const int32_t bias[5] = {INT32_MIN, INT32_MAX, 0, 8, INT32_MIN};
    int16_t output[5] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_vec_mat_mul_result_acc_s8_s16(lhs, rhs, bias, output, 1073741824, 0, 1, 5, 1, 1));
    for (int row = 0; row < 5; ++row)
    {
        TEST_ASSERT_EQUAL_INT16(quantized_reference((int64_t)bias[row] - 1, 1073741824, -32768, 32767), output[row]);
    }
}
