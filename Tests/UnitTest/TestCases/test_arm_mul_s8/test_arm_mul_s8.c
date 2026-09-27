/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "unity.h"
#include <string.h>

#include "../TestData/mul_scalar_s8/test_data.h"
#include "../TestData/mul_ident_s8/test_data.h"
#include "../TestData/mul_broadcast_h_s8/test_data.h"
#include "../TestData/mul_broadcast_w_s8/test_data.h"
#include "../TestData/mul_broadcast_c_s8/test_data.h"
#include "../TestData/mul_broadcast_hc_s8/test_data.h"

#include "../Utils/validate.h"

void mul_scalar_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_SCALAR_S8_LHS_N;
    lhs_dims.h = MUL_SCALAR_S8_LHS_H;
    lhs_dims.w = MUL_SCALAR_S8_LHS_W;
    lhs_dims.c = MUL_SCALAR_S8_LHS_C;

    rhs_dims.n = MUL_SCALAR_S8_RHS_N;
    rhs_dims.h = MUL_SCALAR_S8_RHS_H;
    rhs_dims.w = MUL_SCALAR_S8_RHS_W;
    rhs_dims.c = MUL_SCALAR_S8_RHS_C;

    out_dims.n = MUL_SCALAR_S8_OUTPUT_N;
    out_dims.h = MUL_SCALAR_S8_OUTPUT_H;
    out_dims.w = MUL_SCALAR_S8_OUTPUT_W;
    out_dims.c = MUL_SCALAR_S8_OUTPUT_C;

    const int8_t *lhs = mul_scalar_s8_lhs_input_tensor;
    const int8_t *rhs = mul_scalar_s8_rhs_input_tensor;
    int8_t output[MUL_SCALAR_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_SCALAR_S8_LHS_OFFSET,
        MUL_SCALAR_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_SCALAR_S8_OUTPUT_OFFSET,
        MUL_SCALAR_S8_OUTPUT_MULT,
        MUL_SCALAR_S8_OUTPUT_SHIFT,
        MUL_SCALAR_S8_ACTIVATION_MIN,
        MUL_SCALAR_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_scalar_s8_output_ref, MUL_SCALAR_S8_DST_SIZE));
}


void mul_ident_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_IDENT_S8_LHS_N;
    lhs_dims.h = MUL_IDENT_S8_LHS_H;
    lhs_dims.w = MUL_IDENT_S8_LHS_W;
    lhs_dims.c = MUL_IDENT_S8_LHS_C;

    rhs_dims.n = MUL_IDENT_S8_RHS_N;
    rhs_dims.h = MUL_IDENT_S8_RHS_H;
    rhs_dims.w = MUL_IDENT_S8_RHS_W;
    rhs_dims.c = MUL_IDENT_S8_RHS_C;

    out_dims.n = MUL_IDENT_S8_OUTPUT_N;
    out_dims.h = MUL_IDENT_S8_OUTPUT_H;
    out_dims.w = MUL_IDENT_S8_OUTPUT_W;
    out_dims.c = MUL_IDENT_S8_OUTPUT_C;

    const int8_t *lhs = mul_ident_s8_lhs_input_tensor;
    const int8_t *rhs = mul_ident_s8_rhs_input_tensor;
    int8_t output[MUL_IDENT_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_IDENT_S8_LHS_OFFSET,
        MUL_IDENT_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_IDENT_S8_OUTPUT_OFFSET,
        MUL_IDENT_S8_OUTPUT_MULT,
        MUL_IDENT_S8_OUTPUT_SHIFT,
        MUL_IDENT_S8_ACTIVATION_MIN,
        MUL_IDENT_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_ident_s8_output_ref, MUL_IDENT_S8_DST_SIZE));
}


void mul_broadcast_h_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_H_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_H_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_H_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_H_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_H_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_H_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_H_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_H_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_H_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_H_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_H_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_H_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_h_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_h_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_H_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_H_S8_LHS_OFFSET,
        MUL_BROADCAST_H_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_H_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_H_S8_OUTPUT_MULT,
        MUL_BROADCAST_H_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_H_S8_ACTIVATION_MIN,
        MUL_BROADCAST_H_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_h_s8_output_ref, MUL_BROADCAST_H_S8_DST_SIZE));
}

void mul_broadcast_w_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_W_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_W_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_W_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_W_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_W_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_W_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_W_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_W_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_W_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_W_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_W_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_W_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_w_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_w_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_W_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_W_S8_LHS_OFFSET,
        MUL_BROADCAST_W_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_W_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_W_S8_OUTPUT_MULT,
        MUL_BROADCAST_W_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_W_S8_ACTIVATION_MIN,
        MUL_BROADCAST_W_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_w_s8_output_ref, MUL_BROADCAST_W_S8_DST_SIZE));
}

void mul_broadcast_c_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_C_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_C_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_C_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_C_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_C_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_C_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_C_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_C_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_C_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_C_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_C_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_C_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_c_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_c_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_C_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_C_S8_LHS_OFFSET,
        MUL_BROADCAST_C_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_C_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_C_S8_OUTPUT_MULT,
        MUL_BROADCAST_C_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_C_S8_ACTIVATION_MIN,
        MUL_BROADCAST_C_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_c_s8_output_ref, MUL_BROADCAST_C_S8_DST_SIZE));
}

void mul_broadcast_hc_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_HC_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_HC_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_HC_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_HC_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_HC_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_HC_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_HC_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_HC_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_HC_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_HC_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_HC_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_HC_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_hc_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_hc_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_HC_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_HC_S8_LHS_OFFSET,
        MUL_BROADCAST_HC_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_HC_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_HC_S8_OUTPUT_MULT,
        MUL_BROADCAST_HC_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_HC_S8_ACTIVATION_MIN,
        MUL_BROADCAST_HC_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_hc_s8_output_ref, MUL_BROADCAST_HC_S8_DST_SIZE));
}

/* Regression for the NHWC broadcast walk (issue #336): see the add suite for the mechanism.
 * Identity output requantization and zero input offsets, so the expected output is the plain
 * elementwise product. */
void mul_broadcast_batch_scalar_s8_arm_mul_s8(void)
{
    const int8_t input_1[2] = {10, 20};
    const int8_t input_2[4] = {1, 2, 3, 4};
    const int8_t expected_1_2[8] = {10, 20, 30, 40, 20, 40, 60, 80};
    const int8_t expected_2_1[8] = {10, 20, 30, 40, 20, 40, 60, 80};
    int8_t output[8] = {0};
    int8_t output_reversed[8] = {0};
    const cmsis_nn_dims input_1_dims = {2, 1, 1, 1};
    const cmsis_nn_dims input_2_dims = {1, 2, 1, 2};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_mul_s8(
            input_1, &input_1_dims, input_2, &input_2_dims, 0, 0, output, &output_dims, 0, 1073741824, 1, -128, 127));
    TEST_ASSERT_TRUE(validate(output, expected_1_2, 8));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_mul_s8(input_2,
                                 &input_2_dims,
                                 input_1,
                                 &input_1_dims,
                                 0,
                                 0,
                                 output_reversed,
                                 &output_dims,
                                 0,
                                 1073741824,
                                 1,
                                 -128,
                                 127));
    TEST_ASSERT_TRUE(validate(output_reversed, expected_2_1, 8));
}

/* The walk indexes each operand by its own dims, so shapes that do not broadcast to the output
 * shape are now rejected instead of silently producing a partial result. */
void mul_dims_arg_error_s8_arm_mul_s8(void)
{
    const int8_t input_a[4] = {1, 2, 3, 4};
    const int8_t input_b[4] = {5, 6, 7, 8};
    int8_t output[8] = {0};
    const cmsis_nn_dims dims_2n = {2, 1, 1, 1};
    const cmsis_nn_dims dims_3n = {3, 1, 1, 1};
    const cmsis_nn_dims dims_1h2c = {1, 2, 1, 2};
    const cmsis_nn_dims dims_out = {2, 2, 1, 2};
    const cmsis_nn_dims dims_0h = {2, 0, 1, 1};

    /* n = 2 against n = 3 does not broadcast */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(input_a, &dims_2n, input_b, &dims_3n, 0, 0, output, &dims_3n, 0, 1073741824, 1, -128, 127));
    /* the output shape must be the broadcast shape of the two inputs */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(input_a, &dims_2n, input_b, &dims_1h2c, 0, 0, output, &dims_2n, 0, 1073741824, 1, -128, 127));
    /* a null operand is rejected */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(NULL, &dims_2n, input_b, &dims_1h2c, 0, 0, output, &dims_out, 0, 1073741824, 1, -128, 127));
    /* a non-positive dimension is rejected */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(input_a, &dims_0h, input_b, &dims_0h, 0, 0, output, &dims_0h, 0, 1073741824, 1, -128, 127));
}

/* One operand broadcast along W with matching C, including swapped operands, N/H broadcast and C not a multiple of
   8, compared against a scalar reference. Guard bytes after the output catch writes past the end. */
#define ROWB_MAX (2 * 3 * 240 * 8)
static int8_t rowb_in_full[ROWB_MAX], rowb_in_row[ROWB_MAX], rowb_out[ROWB_MAX + 16], rowb_ref[ROWB_MAX];

static int32_t rowb_size(const cmsis_nn_dims *d) { return d->n * d->h * d->w * d->c; }

static int32_t rowb_index(const cmsis_nn_dims *d, int32_t n, int32_t h, int32_t w, int32_t c)
{
    return (((d->n == 1 ? 0 : n) * d->h + (d->h == 1 ? 0 : h)) * d->w + (d->w == 1 ? 0 : w)) * d->c + c;
}

static void mul_row_broadcast_case(const cmsis_nn_dims full,
                                   const cmsis_nn_dims row,
                                   const cmsis_nn_dims out,
                                   const int32_t swap,
                                   const int32_t off_full,
                                   const int32_t off_row,
                                   const int32_t out_offset,
                                   const int32_t out_mult,
                                   const int32_t out_shift,
                                   const int32_t act_min,
                                   const int32_t act_max)
{
    uint32_t seed = (uint32_t)(full.w * 131 + full.c * 7 + swap);
    for (int32_t i = 0; i < rowb_size(&full); i++)
    {
        seed = seed * 1664525u + 1013904223u;
        rowb_in_full[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < rowb_size(&row); i++)
    {
        seed = seed * 1664525u + 1013904223u;
        rowb_in_row[i] = (int8_t)(seed >> 24);
    }
    for (int32_t n = 0; n < out.n; n++)
        for (int32_t h = 0; h < out.h; h++)
            for (int32_t w = 0; w < out.w; w++)
                for (int32_t c = 0; c < out.c; c++)
                {
                    int32_t r = (rowb_in_full[rowb_index(&full, n, h, w, c)] + off_full) *
                        (rowb_in_row[rowb_index(&row, n, h, w, c)] + off_row);
                    r = arm_nn_requantize(r, out_mult, out_shift) + out_offset;
                    r = r < act_min ? act_min : (r > act_max ? act_max : r);
                    rowb_ref[rowb_index(&out, n, h, w, c)] = (int8_t)r;
                }
    memset(rowb_out, 0x5A, sizeof(rowb_out));
    const arm_cmsis_nn_status status = swap ? arm_mul_s8(rowb_in_row,
                                                         &row,
                                                         rowb_in_full,
                                                         &full,
                                                         off_row,
                                                         off_full,
                                                         rowb_out,
                                                         &out,
                                                         out_offset,
                                                         out_mult,
                                                         out_shift,
                                                         act_min,
                                                         act_max)
                                            : arm_mul_s8(rowb_in_full,
                                                         &full,
                                                         rowb_in_row,
                                                         &row,
                                                         off_full,
                                                         off_row,
                                                         rowb_out,
                                                         &out,
                                                         out_offset,
                                                         out_mult,
                                                         out_shift,
                                                         act_min,
                                                         act_max);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT8_ARRAY(rowb_ref, rowb_out, rowb_size(&out));
    for (int32_t i = 0; i < 16; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, rowb_out[rowb_size(&out) + i]);
    }
}

void mul_row_broadcast_s8_arm_mul_s8(void)
{
    /* {W operand, W == 1 operand, output} */
    const cmsis_nn_dims shapes[][3] = {
        {{1, 1, 240, 8}, {1, 1, 1, 8}, {1, 1, 240, 8}},
        {{1, 1, 17, 3}, {1, 1, 1, 3}, {1, 1, 17, 3}},
        {{2, 3, 5, 12}, {1, 1, 1, 12}, {2, 3, 5, 12}},
        {{2, 3, 5, 17}, {2, 3, 1, 17}, {2, 3, 5, 17}},
        {{1, 1, 5, 12}, {2, 3, 1, 12}, {2, 3, 5, 12}},
        {{1, 2, 9, 40}, {1, 2, 1, 40}, {1, 2, 9, 40}},
        {{1, 1, 2, 2}, {1, 1, 1, 2}, {1, 1, 2, 2}},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    {
        for (int32_t swap = 0; swap < 2; swap++)
        {
            const cmsis_nn_dims *d = shapes[i];
            mul_row_broadcast_case(d[0], d[1], d[2], swap, 128, 128, -128, 1173388748, -7, -128, 127);
            mul_row_broadcast_case(d[0], d[1], d[2], swap, -3, 17, 5, 1518500250, -9, -100, 90);
            mul_row_broadcast_case(d[0], d[1], d[2], swap, 7, -2, -1, 1073741824, 0, -128, 127);
        }
    }
}

