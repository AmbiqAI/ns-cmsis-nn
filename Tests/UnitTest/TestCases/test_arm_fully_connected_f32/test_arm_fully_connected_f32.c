/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <unity.h>

// arm_fully_connected_f32 on top of arm_nn_mat_mult_nt_t_f32: K >= 32 takes the contiguous-K MVE kernel in
// groups of four rhs rows plus a single-row remainder, K < 32 the gather kernel (#417). Values are small
// multiples of 1/8 and 1/4 so every product is a multiple of 1/32 and the f32 reference is exact.
// The f32 kernel result is exact, so the tolerance only absorbs the reference's own rounding.

#define FC_MAX_BATCH 2
#define FC_MAX_K 1024
#define FC_MAX_N 13

static float32_t fc_x[FC_MAX_BATCH * FC_MAX_K];
static float32_t fc_w[FC_MAX_N * FC_MAX_K];
static float32_t fc_bias[FC_MAX_N];
static float32_t fc_y[FC_MAX_BATCH * FC_MAX_N];
static float32_t fc_ref[FC_MAX_BATCH * FC_MAX_N];

// Deterministic hash so rows with K a multiple of the period do not repeat each other.
static uint32_t fc_hash(int32_t i, int32_t seed)
{
    uint32_t h = (uint32_t)i * 2654435761u + (uint32_t)seed * 40503u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

static float32_t fc_input_value(int32_t i, int32_t seed)
{
    return (float32_t)((float32_t)((int32_t)(fc_hash(i, seed) & 15u) - 8) / 8.0f);
}

static float32_t fc_weight_value(int32_t i, int32_t seed)
{
    return (float32_t)((float32_t)((int32_t)(fc_hash(i, seed) & 7u) - 4) / 4.0f);
}

static void fc_f32_case(int32_t batch, int32_t k, int32_t n, int32_t use_bias, float32_t act_min, float32_t act_max)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {batch, 1, 1, k};
    const cmsis_nn_dims filter_dims = {k, 1, 1, n};
    const cmsis_nn_dims bias_dims = {1, 1, 1, n};
    const cmsis_nn_dims output_dims = {batch, 1, 1, n};
    const int32_t seed = k * 7 + n;
    cmsis_nn_fc_params_f32 fc_params;

    TEST_ASSERT_TRUE(batch <= FC_MAX_BATCH && k <= FC_MAX_K && n <= FC_MAX_N);
    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = (float32_t)act_min;
    fc_params.activation.max = (float32_t)act_max;
    fc_params.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;

    for (int32_t i = 0; i < batch * k; ++i)
    {
        fc_x[i] = fc_input_value(i, seed);
    }
    for (int32_t i = 0; i < n * k; ++i)
    {
        fc_w[i] = fc_weight_value(i, seed + 1);
    }
    for (int32_t i = 0; i < n; ++i)
    {
        fc_bias[i] = fc_input_value(i, seed + 2);
    }
    for (int32_t i = 0; i < batch * n; ++i)
    {
        fc_y[i] = (float32_t)0.0f;
    }

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t col = 0; col < n; ++col)
        {
            float32_t acc = use_bias ? (float32_t)fc_bias[col] : 0.0f;
            for (int32_t i = 0; i < k; ++i)
            {
                acc += (float32_t)fc_x[b * k + i] * (float32_t)fc_w[col * k + i];
            }
            fc_ref[b * n + col] = fminf(fmaxf(acc, act_min), act_max);
        }
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f32(&ctx,
                                              &fc_params,
                                              &input_dims,
                                              fc_x,
                                              &filter_dims,
                                              fc_w,
                                              &bias_dims,
                                              use_bias ? fc_bias : NULL,
                                              &output_dims,
                                              fc_y,
                                              ARM_NN_LAYOUT_NHWC));

    for (int32_t i = 0; i < batch * n; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, fc_ref[i], (float32_t)fc_y[i]);
    }
}

// The KWS classifier layer: three groups of four rhs rows, no remainder.
void fully_connected_1024_to_12_f32(void) { fc_f32_case(1, 1024, 12, 1, -1.0e4f, 1.0e4f); }

// One remainder row after the groups of four; batch 2 with and without bias.
void fully_connected_1024_to_13_batch2_f32(void)
{
    fc_f32_case(2, 1024, 13, 1, -1.0e4f, 1.0e4f);
    fc_f32_case(2, 1024, 13, 0, -1.0e4f, 1.0e4f);
}

// K just at and past the contiguous threshold with a one-element and a seven-element vector tail.
void fully_connected_k33_k39_n5_f32(void)
{
    fc_f32_case(1, 33, 5, 1, -1.0e4f, 1.0e4f);
    fc_f32_case(1, 39, 5, 1, -1.0e4f, 1.0e4f);
}

// Tight clamp on a contiguous-K shape: both bounds must bite.
void fully_connected_k39_n5_clamped_f32(void) { fc_f32_case(1, 39, 5, 1, -1.5f, 1.5f); }

// K just below the threshold: the gather kernel, one full 8-row block plus a single-row remainder.
void fully_connected_k31_n9_f32(void)
{
    fc_f32_case(1, 31, 9, 1, -1.0e4f, 1.0e4f);
    fc_f32_case(2, 31, 9, 0, -1.0e4f, 1.0e4f);
}

// Every output row of a product is the single-row product of that row: the rows a block computes
// together (four scalar accumulation chains, the Helium gather and contiguous blocks) and the rows
// the remainder loop computes one at a time are the same products summed in the same order from
// 0.0f, then added to the bias. Values are not multiples of a power of two here, so a reordered sum
// would show in the low bits. Where the compiler contracts a*b+c the same way in every loop (GCC's
// -ffp-contract=fast, its default in GNU mode, and off, its default under -std=c11; verified on the
// Cortex-M55 toolchain) the rows are bit for bit; Clang's default -ffp-contract=on fuses one loop
// shape and not another, and then a row differs by the one rounding a fused multiply-add saves, which
// is the only difference the tolerance admits (a reordered sum of these values is far outside it).
// The shapes reach the scalar four-row block with a three-row remainder (n = 7), the Helium gather
// block (k = 9) and the contiguous block (k = 37), and a bias both present and absent.
static void fc_f32_rows_match_single_rows(int32_t k, int32_t n, int32_t use_bias)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {1, 1, 1, k};
    const cmsis_nn_dims filter_dims = {k, 1, 1, n};
    const cmsis_nn_dims bias_dims = {1, 1, 1, n};
    const cmsis_nn_dims output_dims = {1, 1, 1, n};
    const cmsis_nn_dims one_filter_dims = {k, 1, 1, 1};
    const cmsis_nn_dims one_bias_dims = {1, 1, 1, 1};
    const cmsis_nn_dims one_output_dims = {1, 1, 1, 1};
    cmsis_nn_fc_params_f32 fc_params;
    uint32_t whole[FC_MAX_N];
    uint32_t single[FC_MAX_N];

    TEST_ASSERT_TRUE(k <= FC_MAX_K && n <= FC_MAX_N);
    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = -FLT_MAX;
    fc_params.activation.max = FLT_MAX;
    fc_params.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;
    for (int32_t i = 0; i < k; ++i)
    {
        fc_x[i] = 0.37f * (float32_t)((int32_t)(fc_hash(i, 11) & 31u) - 16) + 0.013f;
    }
    for (int32_t i = 0; i < n * k; ++i)
    {
        fc_w[i] = 0.29f * (float32_t)((int32_t)(fc_hash(i, 13) & 31u) - 15) - 0.007f;
    }
    for (int32_t i = 0; i < n; ++i)
    {
        fc_bias[i] = 0.61f * (float32_t)((int32_t)(fc_hash(i, 17) & 15u) - 7);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f32(&ctx,
                                              &fc_params,
                                              &input_dims,
                                              fc_x,
                                              &filter_dims,
                                              fc_w,
                                              &bias_dims,
                                              use_bias ? fc_bias : NULL,
                                              &output_dims,
                                              fc_y,
                                              ARM_NN_LAYOUT_NHWC));
    memcpy(whole, fc_y, (size_t)n * sizeof(uint32_t));
    for (int32_t row = 0; row < n; ++row)
    {
        float32_t one;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_fully_connected_f32(&ctx,
                                                  &fc_params,
                                                  &input_dims,
                                                  fc_x,
                                                  &one_filter_dims,
                                                  fc_w + (size_t)row * k,
                                                  &one_bias_dims,
                                                  use_bias ? fc_bias + row : NULL,
                                                  &one_output_dims,
                                                  &one,
                                                  ARM_NN_LAYOUT_NHWC));
        memcpy(&single[row], &one, sizeof(uint32_t));
    }
    for (int32_t row = 0; row < n; ++row)
    {
        float32_t expected;
        float32_t got;
        memcpy(&expected, &single[row], sizeof(expected));
        memcpy(&got, &whole[row], sizeof(got));
        TEST_ASSERT_FLOAT_WITHIN(fabsf(expected) * 1.0e-6f + 1.0e-6f, expected, got);
    }
#if defined(__GNUC__) && !defined(__clang__)
    TEST_ASSERT_EQUAL_HEX32_ARRAY(single, whole, n);
#endif
}

void fully_connected_rows_match_single_rows_f32(void)
{
    fc_f32_rows_match_single_rows(9, 7, 1);
    fc_f32_rows_match_single_rows(9, 7, 0);
    fc_f32_rows_match_single_rows(37, 7, 1);
    fc_f32_rows_match_single_rows(64, 10, 1);
    fc_f32_rows_match_single_rows(9, 4, 1);
}
