/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <string.h>
#include <unity.h>

// arm_fully_connected_f16 on top of arm_nn_mat_mult_nt_t_f16: K >= 32 takes the contiguous-K MVE kernel in
// groups of four rhs rows plus a single-row remainder, K < 32 the gather kernel (#417). Values are small
// multiples of 1/8 and 1/4 so every product is a multiple of 1/32 and the f32 reference is exact.
// f16 accumulation is exact while every partial sum stays a multiple of 1/32 below 2^6, which these
// magnitudes do; the tolerance is the batch_matmul_f16 class in case a partial sum crosses that.

#define FC_MAX_BATCH 2
#define FC_MAX_K 1024
#define FC_MAX_N 13

static float16_t fc_x[FC_MAX_BATCH * FC_MAX_K];
static float16_t fc_w[FC_MAX_N * FC_MAX_K];
static float16_t fc_bias[FC_MAX_N];
static float16_t fc_y[FC_MAX_BATCH * FC_MAX_N];
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

static float16_t fc_input_value(int32_t i, int32_t seed)
{
    return (float16_t)((float32_t)((int32_t)(fc_hash(i, seed) & 15u) - 8) / 8.0f);
}

static float16_t fc_weight_value(int32_t i, int32_t seed)
{
    return (float16_t)((float32_t)((int32_t)(fc_hash(i, seed) & 7u) - 4) / 4.0f);
}

static void fc_f16_case(int32_t batch, int32_t k, int32_t n, int32_t use_bias, float32_t act_min, float32_t act_max)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {batch, 1, 1, k};
    const cmsis_nn_dims filter_dims = {k, 1, 1, n};
    const cmsis_nn_dims bias_dims = {1, 1, 1, n};
    const cmsis_nn_dims output_dims = {batch, 1, 1, n};
    const int32_t seed = k * 7 + n;
    cmsis_nn_fc_params_f16 fc_params;

    TEST_ASSERT_TRUE(batch <= FC_MAX_BATCH && k <= FC_MAX_K && n <= FC_MAX_N);
    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = (float16_t)act_min;
    fc_params.activation.max = (float16_t)act_max;
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
        fc_y[i] = (float16_t)0.0f;
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
                      arm_fully_connected_f16(&ctx,
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
        TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, fc_ref[i], (float32_t)fc_y[i]);
    }
}

// The KWS classifier layer: three groups of four rhs rows, no remainder.
void fully_connected_1024_to_12_f16(void) { fc_f16_case(1, 1024, 12, 1, -1.0e4f, 1.0e4f); }

// One remainder row after the groups of four; batch 2 with and without bias.
void fully_connected_1024_to_13_batch2_f16(void)
{
    fc_f16_case(2, 1024, 13, 1, -1.0e4f, 1.0e4f);
    fc_f16_case(2, 1024, 13, 0, -1.0e4f, 1.0e4f);
}

// K just at and past the contiguous threshold with a one-element and a seven-element vector tail.
void fully_connected_k33_k39_n5_f16(void)
{
    fc_f16_case(1, 33, 5, 1, -1.0e4f, 1.0e4f);
    fc_f16_case(1, 39, 5, 1, -1.0e4f, 1.0e4f);
}

// Tight clamp on a contiguous-K shape: both bounds must bite.
void fully_connected_k39_n5_clamped_f16(void) { fc_f16_case(1, 39, 5, 1, -1.5f, 1.5f); }

// K just below the threshold: the gather kernel, one full 8-row block plus a single-row remainder.
void fully_connected_k31_n9_f16(void)
{
    fc_f16_case(1, 31, 9, 1, -1.0e4f, 1.0e4f);
    fc_f16_case(2, 31, 9, 0, -1.0e4f, 1.0e4f);
}

#define PRECISE_MAX_ROWS 5
#define PRECISE_MAX_K 128
#define PRECISE_MAX_OUTPUTS 9
static float16_t precise_x[PRECISE_MAX_ROWS * PRECISE_MAX_K];
static float16_t precise_w[PRECISE_MAX_OUTPUTS * PRECISE_MAX_K];
static float16_t precise_packed[16 * PRECISE_MAX_K];
static float16_t precise_bias[PRECISE_MAX_OUTPUTS];
static float16_t precise_y[PRECISE_MAX_ROWS * (PRECISE_MAX_OUTPUTS + 3)];
static float16_t precise_legacy[PRECISE_MAX_ROWS * PRECISE_MAX_OUTPUTS];

static uint16_t precise_bits(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void precise_pack(int32_t k, int32_t outputs)
{
    memset(precise_packed, 0, sizeof(precise_packed));
    for (int32_t col = 0; col < outputs; ++col)
    {
        for (int32_t tap = 0; tap < k; ++tap)
        {
            precise_packed[(col / 8) * k * 8 + tap * 8 + col % 8] = precise_w[col * k + tap];
        }
    }
}

static void precise_case(int32_t rows, int32_t k, int32_t outputs, bool bias, float16_t lo, float16_t hi)
{
    const int32_t stride = outputs + 3;
    for (int32_t i = 0; i < rows * k; ++i)
    {
        precise_x[i] = (float16_t)((i % 7 - 3) / 8.0f);
    }
    for (int32_t col = 0; col < outputs; ++col)
    {
        precise_bias[col] = (float16_t)((col % 5 - 2) / 8.0f);
        for (int32_t tap = 0; tap < k; ++tap)
        {
            precise_w[col * k + tap] = (float16_t)(((col + tap) % 5 - 2) / 4.0f);
        }
    }
    precise_pack(k, outputs);
    for (int32_t i = 0; i < rows * stride; ++i)
    {
        precise_y[i] = (float16_t)42;
    }
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mat_mult_nt_n_packed_f16_precise(
            precise_x, precise_packed, bias ? precise_bias : NULL, precise_y, rows, outputs, k, stride, lo, hi));
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t col = 0; col < outputs; ++col)
        {
            float32_t ref = (float32_t)precise_bias[col];
            uint32_t bias_bits;
            memcpy(&bias_bits, &ref, sizeof(bias_bits));
            bias_bits &= 0u - (uint32_t)bias;
            memcpy(&ref, &bias_bits, sizeof(ref));
            for (int32_t tap = 0; tap < k; ++tap)
            {
                ref += (float32_t)precise_x[row * k + tap] * (float32_t)precise_w[col * k + tap];
            }
            if (ref < (float32_t)lo)
                ref = (float32_t)lo;
            if (ref > (float32_t)hi)
                ref = (float32_t)hi;
            /* Every term is a multiple of 1/32 and partial magnitudes stay below 32: these sums are exact. */
            TEST_ASSERT_FLOAT_WITHIN(0.0f, ref, (float32_t)precise_y[row * stride + col]);
        }
        for (int32_t col = outputs; col < stride; ++col)
        {
            TEST_ASSERT_EQUAL_UINT16(precise_bits((float16_t)42), precise_bits(precise_y[row * stride + col]));
        }
    }
    if (k <= 32)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_mat_mult_nt_n_packed_f16(precise_x,
                                                          precise_packed,
                                                          bias ? precise_bias : NULL,
                                                          precise_legacy,
                                                          rows,
                                                          outputs,
                                                          k,
                                                          outputs,
                                                          lo,
                                                          hi));
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_nn_mat_mult_nt_n_packed_f16_acc16(
                precise_x, precise_packed, bias ? precise_bias : NULL, precise_y, rows, outputs, k, outputs, lo, hi));
        for (int32_t i = 0; i < rows * outputs; ++i)
        {
            TEST_ASSERT_EQUAL_UINT16(precise_bits(precise_legacy[i]), precise_bits(precise_y[i]));
        }
    }
}

void fully_connected_packed_precise_contract_f16(void)
{
    precise_case(1, 1, 1, false, (float16_t)-1, (float16_t)1);
    precise_case(1, 3, 7, true, (float16_t)-1, (float16_t)1);
    precise_case(4, 31, 9, false, (float16_t)-1, (float16_t)1);
    precise_case(5, 32, 9, true, (float16_t)-1, (float16_t)1);
    precise_case(1, 33, 1, false, (float16_t)-1, (float16_t)1);
    precise_case(4, 128, 9, true, (float16_t)-0.25f, (float16_t)0.25f);
}

void fully_connected_packed_precise_bias_f16(void)
{
    for (int32_t tap = 0; tap < 32; ++tap)
    {
        precise_x[tap] = (float16_t)64;
        if (tap < 16)
            precise_w[tap] = (float16_t)64;
        else
            precise_w[tap] = (float16_t)-64;
    }
    precise_bias[0] = (float16_t)1;
    precise_pack(32, 1);
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mat_mult_nt_n_packed_f16_precise(
            precise_x, precise_packed, precise_bias, precise_y, 1, 1, 32, 1, (float16_t)-65504, (float16_t)65504));
    TEST_ASSERT_EQUAL_UINT16(precise_bits((float16_t)1), precise_bits(precise_y[0]));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mat_mult_nt_n_packed_f16(
            precise_x, precise_packed, precise_bias, precise_legacy, 1, 1, 32, 1, (float16_t)-65504, (float16_t)65504));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mat_mult_nt_n_packed_f16_acc16(
            precise_x, precise_packed, precise_bias, precise_y, 1, 1, 32, 1, (float16_t)-65504, (float16_t)65504));
    TEST_ASSERT_EQUAL_UINT16(precise_bits(precise_legacy[0]), precise_bits(precise_y[0]));

    /* Four-tap partials are exactly +/-32768; eight taps overflow. Cover four rows and the tail. */
    for (int32_t i = 0; i < 5 * 16; ++i)
    {
        precise_x[i] = (float16_t)128;
    }
    for (int32_t tap = 0; tap < 16; ++tap)
    {
        if (tap < 8)
            precise_w[tap] = (float16_t)64;
        else
            precise_w[tap] = (float16_t)-64;
    }
    precise_pack(16, 1);
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mat_mult_nt_n_packed_f16_precise(
            precise_x, precise_packed, precise_bias, precise_y, 5, 1, 16, 1, (float16_t)-65504, (float16_t)65504));
    for (int32_t row = 0; row < 5; ++row)
    {
        TEST_ASSERT_EQUAL_UINT16(0x3c00, precise_bits(precise_y[row]));
    }
}

void fully_connected_packed_precise_nonfinite_f16(void)
{
    const uint16_t nan_bits = 0x7e55;
    memcpy(precise_x, &nan_bits, sizeof(nan_bits));
    for (int32_t col = 0; col < 3; ++col)
        precise_w[col] = (float16_t)0.5f;
    precise_pack(1, 3);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, precise_packed, NULL, precise_y, 1, 3, 1, 3, (float16_t)-65504, (float16_t)65504));
    for (int32_t col = 0; col < 3; ++col)
    {
        const uint16_t bits = precise_bits(precise_y[col]);
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        /* The documented MVE maxNum/minNum clamp suppresses a NaN to the lower bound. */
        TEST_ASSERT_EQUAL_UINT16(precise_bits((float16_t)-65504), bits);
#else
        TEST_ASSERT_TRUE((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0);
#endif
    }
}

void fully_connected_packed_precise_invalid_f16(void)
{
    precise_x[0] = (float16_t)0.5f;
    precise_packed[0] = (float16_t)0.5f;
    precise_y[0] = (float16_t)42;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          NULL, precise_packed, NULL, precise_y, 1, 1, 1, 1, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, NULL, NULL, precise_y, 1, 1, 1, 1, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, precise_packed, NULL, NULL, 1, 1, 1, 1, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, precise_packed, NULL, precise_y, 0, 1, 1, 1, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, precise_packed, NULL, precise_y, 1, 1, -1, 1, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, precise_packed, NULL, precise_y, 1, 1, 1, 0, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL_UINT16(precise_bits((float16_t)42), precise_bits(precise_y[0]));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mat_mult_nt_n_packed_f16_precise(
                          precise_x, precise_packed, NULL, precise_y, 1, 1, 1, INT32_MAX, (float16_t)-1, (float16_t)1));
    TEST_ASSERT_EQUAL_UINT16(precise_bits((float16_t)0.25f), precise_bits(precise_y[0]));
}
