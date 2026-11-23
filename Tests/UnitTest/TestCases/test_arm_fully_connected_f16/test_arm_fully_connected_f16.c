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
#include <stdio.h>
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
        /* NaN on every leg (ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION). This case used to expect -65504 on the MVE
         * leg, where the maxNum/minNum clamp suppressed the NaN to the lower bound while the scalar leg passed it
         * through; the products now restore the NaN lanes, so both legs agree. */
        TEST_ASSERT_TRUE((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0);
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

// One NaN rule in the float matmul products, float16 twin of the float32 suite's cases: a NaN accumulator, from a
// NaN operand, +inf * 0 or +inf + -inf, is NaN at that output on every leg; an infinite accumulator saturates to the
// activation bound; every other output keeps its bits. Before the rule the Helium blocks resolved a NaN to
// activation.min (vmaxnm/vminnm), the row loops and the contiguous rows to activation.max (arm_nn_clamp_f16h, and
// vminnm/vmaxnm in that order on the folded rows), and only the packed product's scalar leg passed it through.
//
// Each shape reaches named routes of arm_nn_mat_mult_nt_t_f16 / arm_nn_mat_mult_nt_n_packed_f16 on the Helium leg,
// through arm_fully_connected_f16 and arm_fully_connected_f16_acc16; the scalar leg (host, cortex-m55+nomve.fp)
// runs every shape through its row loop:
//   standard, K = 5, N = 29: the dual block (column 3), the eight-row block (18), the four-row block (25), the
//   remainder (28);
//   standard, K = 32, N = 5: the contiguous rows in float16 lanes (column 1), the remainder (4);
//   standard, K = 37 and K = 264, N = 5: the default entry's folded contiguous rows (column 1; one block and several),
//   the folded remainder (4); the _acc16 entry takes the float16-lane contiguous rows;
//   packed, K = 5 and K = 37, N = 13, batch 5: the four-row block (rows 0-3) and the single row (row 4), a full
//   column block (column 2) and the partial one (column 12).
// Specials are rebuilt from bits held in volatiles, and every comparison is on bits: no float16 select is written
// (GCC PR target/118460).

#define FC16_NAN_MAX_BATCH 5
#define FC16_NAN_MAX_K 264
#define FC16_NAN_MAX_N 29
#define FC16_NAN_BLOCK 8
#define FC16_NAN_ENTRIES 2

typedef enum
{
    FC16_NAN_INPUT = 0,
    FC16_NAN_WEIGHT,
    FC16_NAN_BIAS,
    FC16_NAN_INF_TIMES_ZERO,
    FC16_NAN_INF_MINUS_INF
} fc16_nan_kind;

typedef enum
{
    FC16_NAN_SAME = 0,
    FC16_NAN_EXPECT_NAN,
    FC16_NAN_EXPECT_MAX,
    FC16_NAN_EXPECT_MIN
} fc16_nan_expect;

typedef struct
{
    int32_t batch;
    int32_t k;
    int32_t n;
    int32_t packed;
    int32_t cols[4];
    int32_t n_cols;
    int32_t rows[2];
    int32_t n_rows;
} fc16_nan_shape;

static const fc16_nan_shape fc16_nan_shapes[] = {
    {2, 5, 29, 0, {3, 18, 25, 28}, 4, {1, 0}, 1},
    {2, 32, 5, 0, {1, 4, 0, 0}, 2, {1, 0}, 1},
    {2, 37, 5, 0, {1, 4, 0, 0}, 2, {1, 0}, 1},
    {2, 264, 5, 0, {1, 4, 0, 0}, 2, {1, 0}, 1},
    {5, 5, 13, 1, {2, 12, 0, 0}, 2, {1, 4}, 2},
    {5, 37, 13, 1, {2, 12, 0, 0}, 2, {1, 4}, 2},
};

static float16_t fc16_nan_x[FC16_NAN_MAX_BATCH * FC16_NAN_MAX_K];
static float16_t fc16_nan_w[FC16_NAN_MAX_N * FC16_NAN_MAX_K];
static float16_t
    fc16_nan_packed[((FC16_NAN_MAX_N + FC16_NAN_BLOCK - 1) / FC16_NAN_BLOCK) * FC16_NAN_BLOCK * FC16_NAN_MAX_K];
static float16_t fc16_nan_bias[FC16_NAN_MAX_N];
static float16_t fc16_nan_clean[FC16_NAN_MAX_BATCH * FC16_NAN_MAX_N];
static float16_t fc16_nan_y[FC16_NAN_MAX_BATCH * FC16_NAN_MAX_N];
static fc16_nan_expect fc16_nan_want[FC16_NAN_MAX_BATCH * FC16_NAN_MAX_N];

static float16_t fc16_nan_from_bits(volatile const uint16_t *bits)
{
    const uint16_t b = *bits;
    float16_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static float16_t fc16_nan_make(uint16_t bits)
{
    float16_t x;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static uint16_t fc16_nan_bits(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int32_t fc16_nan_is_nan(float16_t x) { return (fc16_nan_bits(x) & 0x7FFFu) > 0x7C00u; }

// Non-zero dyadic values of both signs, so every finite sum is exact in float16 and every weight has a sign.
static float16_t fc16_nan_value(int32_t i, int32_t seed, float32_t scale)
{
    const uint32_t h = fc_hash(i, seed);
    const float16_t magnitude = (float16_t)((float32_t)((int32_t)(h & 7u) + 1) * scale);
    return fc16_nan_make((uint16_t)(fc16_nan_bits(magnitude) | ((h & 8u) ? 0x8000u : 0u)));
}

static void fc16_nan_run(const fc16_nan_shape *s, int32_t entry, uint16_t lo, uint16_t hi, float16_t *out)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {s->batch, 1, 1, s->k};
    const cmsis_nn_dims filter_dims = {s->k, 1, 1, s->n};
    const cmsis_nn_dims bias_dims = {1, 1, 1, s->n};
    const cmsis_nn_dims output_dims = {s->batch, 1, 1, s->n};
    cmsis_nn_fc_params_f16 fc_params;

    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = fc16_nan_make(lo);
    fc_params.activation.max = fc16_nan_make(hi);
    fc_params.weight_format = s->packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;

    if (s->packed)
    {
        // [N / 8][K][8]: column c's tap k at ((c / 8) * K + k) * 8 + c % 8; the partial block's spare lanes are zero.
        memset(fc16_nan_packed, 0, sizeof(fc16_nan_packed));
        for (int32_t c = 0; c < s->n; ++c)
        {
            for (int32_t k = 0; k < s->k; ++k)
            {
                fc16_nan_packed[((c / FC16_NAN_BLOCK) * s->k + k) * FC16_NAN_BLOCK + c % FC16_NAN_BLOCK] =
                    fc16_nan_w[c * s->k + k];
            }
        }
    }
    memset(out, 0, sizeof(float16_t) * (size_t)(s->batch * s->n));

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        (entry ? arm_fully_connected_f16_acc16 : arm_fully_connected_f16)(&ctx,
                                                                          &fc_params,
                                                                          &input_dims,
                                                                          fc16_nan_x,
                                                                          &filter_dims,
                                                                          s->packed ? fc16_nan_packed : fc16_nan_w,
                                                                          &bias_dims,
                                                                          fc16_nan_bias,
                                                                          &output_dims,
                                                                          out,
                                                                          ARM_NN_LAYOUT_NHWC));
}

static void fc16_nan_case(fc16_nan_kind kind, uint16_t lo, uint16_t hi)
{
    volatile uint16_t nan_bits = 0x7E00u;
    volatile uint16_t inf_bits = 0x7C00u;
    volatile uint16_t ninf_bits = 0xFC00u;
    const float16_t nan_v = fc16_nan_from_bits(&nan_bits);
    const float16_t inf_v = fc16_nan_from_bits(&inf_bits);
    const float16_t ninf_v = fc16_nan_from_bits(&ninf_bits);

    for (size_t si = 0; si < sizeof(fc16_nan_shapes) / sizeof(fc16_nan_shapes[0]); ++si)
    {
        for (int32_t entry = 0; entry < FC16_NAN_ENTRIES; ++entry)
        {
            const fc16_nan_shape *s = &fc16_nan_shapes[si];
            const int32_t tap = s->k - 1;
            const int32_t seed = s->k * 7 + s->n;

            for (int32_t i = 0; i < s->batch * s->k; ++i)
            {
                fc16_nan_x[i] = fc16_nan_value(i, seed, 0.125f);
            }
            for (int32_t i = 0; i < s->n * s->k; ++i)
            {
                fc16_nan_w[i] = fc16_nan_value(i, seed + 1, 0.25f);
            }
            for (int32_t i = 0; i < s->n; ++i)
            {
                fc16_nan_bias[i] = fc16_nan_value(i, seed + 2, 0.125f);
            }
            for (int32_t i = 0; i < s->batch * s->n; ++i)
            {
                fc16_nan_want[i] = FC16_NAN_SAME;
            }

            // Finite set-up first, so the clean run already holds it.
            if (kind == FC16_NAN_INF_TIMES_ZERO)
            {
                for (int32_t j = 0; j < s->n_cols; ++j)
                {
                    fc16_nan_w[s->cols[j] * s->k + tap] = fc16_nan_make(0u);
                }
            }
            if (kind == FC16_NAN_INF_MINUS_INF)
            {
                // Features 0 and `tap` share a sign in every row, so +inf and -inf weights there give products of
                // opposite signs.
                for (int32_t r = 0; r < s->batch; ++r)
                {
                    const uint16_t first = fc16_nan_bits(fc16_nan_x[r * s->k]);
                    const uint16_t last = fc16_nan_bits(fc16_nan_x[r * s->k + tap]);
                    fc16_nan_x[r * s->k + tap] = fc16_nan_make((uint16_t)((last & 0x7FFFu) | (first & 0x8000u)));
                }
            }
            fc16_nan_run(s, entry, lo, hi, fc16_nan_clean);

            switch (kind)
            {
            case FC16_NAN_INPUT:
                for (int32_t j = 0; j < s->n_rows; ++j)
                {
                    fc16_nan_x[s->rows[j] * s->k + tap] = nan_v;
                    for (int32_t c = 0; c < s->n; ++c)
                    {
                        fc16_nan_want[s->rows[j] * s->n + c] = FC16_NAN_EXPECT_NAN;
                    }
                }
                break;
            case FC16_NAN_WEIGHT:
            case FC16_NAN_BIAS:
            case FC16_NAN_INF_MINUS_INF:
                for (int32_t j = 0; j < s->n_cols; ++j)
                {
                    const int32_t c = s->cols[j];
                    if (kind == FC16_NAN_WEIGHT)
                    {
                        fc16_nan_w[c * s->k + tap] = nan_v;
                    }
                    else if (kind == FC16_NAN_BIAS)
                    {
                        fc16_nan_bias[c] = nan_v;
                    }
                    else
                    {
                        // This column's dot meets an infinity of each sign: inf - inf.
                        fc16_nan_w[c * s->k + 0] = inf_v;
                        fc16_nan_w[c * s->k + tap] = ninf_v;
                    }
                    for (int32_t r = 0; r < s->batch; ++r)
                    {
                        fc16_nan_want[r * s->n + c] = FC16_NAN_EXPECT_NAN;
                    }
                }
                break;
            case FC16_NAN_INF_TIMES_ZERO:
                // +inf meets a zero weight in the target columns (NaN) and a non-zero one elsewhere (+-inf,
                // saturated).
                for (int32_t j = 0; j < s->n_rows; ++j)
                {
                    const int32_t r = s->rows[j];
                    fc16_nan_x[r * s->k + tap] = inf_v;
                    for (int32_t c = 0; c < s->n; ++c)
                    {
                        const uint16_t w = fc16_nan_bits(fc16_nan_w[c * s->k + tap]);
                        if ((w & 0x7FFFu) == 0u)
                        {
                            fc16_nan_want[r * s->n + c] = FC16_NAN_EXPECT_NAN;
                        }
                        else if (w & 0x8000u)
                        {
                            fc16_nan_want[r * s->n + c] = FC16_NAN_EXPECT_MIN;
                        }
                        else
                        {
                            fc16_nan_want[r * s->n + c] = FC16_NAN_EXPECT_MAX;
                        }
                    }
                }
                break;
            }

            fc16_nan_run(s, entry, lo, hi, fc16_nan_y);

            for (int32_t i = 0; i < s->batch * s->n; ++i)
            {
                const uint16_t got = fc16_nan_bits(fc16_nan_y[i]);
                char msg[112];
                snprintf(msg,
                         sizeof(msg),
                         "shape %d entry %d output %d (row %d, column %d)",
                         (int)si,
                         (int)entry,
                         (int)i,
                         (int)(i / s->n),
                         (int)(i % s->n));
                switch (fc16_nan_want[i])
                {
                case FC16_NAN_EXPECT_NAN:
                    TEST_ASSERT_TRUE_MESSAGE(fc16_nan_is_nan(fc16_nan_y[i]), msg);
                    break;
                case FC16_NAN_EXPECT_MAX:
                    TEST_ASSERT_EQUAL_HEX16_MESSAGE(hi, got, msg);
                    break;
                case FC16_NAN_EXPECT_MIN:
                    TEST_ASSERT_EQUAL_HEX16_MESSAGE(lo, got, msg);
                    break;
                default:
                    TEST_ASSERT_EQUAL_HEX16_MESSAGE(fc16_nan_bits(fc16_nan_clean[i]), got, msg);
                    break;
                }
            }
        }
    }
}

// Each kind under the identity bounds (+-65504) and under a RELU6 bound, whose zero lower bound a NaN must pass too.
static void fc16_nan_kind_both_bounds(fc16_nan_kind kind)
{
    fc16_nan_case(kind, 0xFBFFu, 0x7BFFu);
    fc16_nan_case(kind, 0x0000u, 0x4600u);
}

void fully_connected_nan_input_f16(void) { fc16_nan_kind_both_bounds(FC16_NAN_INPUT); }

void fully_connected_nan_weight_f16(void) { fc16_nan_kind_both_bounds(FC16_NAN_WEIGHT); }

void fully_connected_nan_bias_f16(void) { fc16_nan_kind_both_bounds(FC16_NAN_BIAS); }

void fully_connected_inf_times_zero_f16(void) { fc16_nan_kind_both_bounds(FC16_NAN_INF_TIMES_ZERO); }

void fully_connected_inf_minus_inf_f16(void) { fc16_nan_kind_both_bounds(FC16_NAN_INF_MINUS_INF); }

// One row, one class, on every backend: helia-ml's ADR 0074 ("Exactness is two layers") holds class decisions exact
// across backends. The float16 twin of the row of finding 1 of the K3 rubric (one NaN rule in the float matmul
// products): x = [+inf, 1] against class rows [0, 1] [1, 0] [-1, 0] [-1, 1], no bias, bounds +-65504. Class 0 is
// 0 * inf = NaN, class 1 +inf, classes 2 and 3 -inf. Before the rule the Helium four-row block wrote -65504 for
// class 0 and answered class 1, the scalar row loop (arm_nn_clamp_f16h) wrote +65504 and answered class 0, and the
// three-class form took the row loop on both legs and answered class 0. With the NaN passed through,
// arm_argmax_f16 skips it and every leg answers class 1.
void fully_connected_nan_row_one_class_f16(void)
{
    volatile uint16_t inf_bits = 0x7C00u;
    const float16_t inf_v = fc16_nan_from_bits(&inf_bits);
    const float16_t one = (float16_t)1.0f;
    const float16_t zero = fc16_nan_make(0x0000u);
    const float16_t minus_one = (float16_t)-1.0f;
    const float16_t x[2] = {inf_v, one};
    const float16_t w[8] = {zero, one, one, zero, minus_one, zero, minus_one, one};
    const cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_fc_params_f16 fc_params;

    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = fc16_nan_make(0xFBFFu);
    fc_params.activation.max = fc16_nan_make(0x7BFFu);
    fc_params.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;

    for (int32_t classes = 4; classes >= 3; --classes)
    {
        const cmsis_nn_dims input_dims = {1, 1, 1, 2};
        const cmsis_nn_dims filter_dims = {2, 1, 1, classes};
        const cmsis_nn_dims bias_dims = {1, 1, 1, classes};
        const cmsis_nn_dims output_dims = {1, 1, 1, classes};
        const cmsis_nn_dims score_dims = {1, 1, 1, classes};
        float16_t scores[4];
        int32_t label = -1;

        memset(scores, 0, sizeof(scores));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_fully_connected_f16(&ctx,
                                                  &fc_params,
                                                  &input_dims,
                                                  x,
                                                  &filter_dims,
                                                  w,
                                                  &bias_dims,
                                                  NULL,
                                                  &output_dims,
                                                  scores,
                                                  ARM_NN_LAYOUT_NHWC));
        TEST_ASSERT_TRUE_MESSAGE(fc16_nan_is_nan(scores[0]), "class 0 is 0 * inf");
        TEST_ASSERT_EQUAL_HEX16(0x7BFFu, fc16_nan_bits(scores[1]));
        TEST_ASSERT_EQUAL_HEX16(0xFBFFu, fc16_nan_bits(scores[2]));
        if (classes == 4)
        {
            TEST_ASSERT_EQUAL_HEX16(0xFBFFu, fc16_nan_bits(scores[3]));
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_argmax_f16(scores, &score_dims, 3, &label));
        TEST_ASSERT_EQUAL_INT32(1, label);
    }
}

// A consumer can tell which rule its heliaCORE applies.
void fully_connected_nan_propagation_macro_f16(void)
{
#if defined(ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION) && (ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION == 1)
    TEST_PASS();
#else
    TEST_FAIL_MESSAGE("ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION is not defined to 1");
#endif
}
