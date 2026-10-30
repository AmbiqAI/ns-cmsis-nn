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
#include <stdio.h>
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

// One NaN rule in the float matmul products: a NaN accumulator, from a NaN operand, +inf * 0 or +inf + -inf, is NaN
// at that output on every leg; an infinite accumulator saturates to the activation bound; every other output keeps
// its bits. Before the rule the Helium blocks resolved a NaN to activation.min (vmaxnm/vminnm) and the scalar rows
// to activation.max (ARM_NN_CLAMP), so one row could decide two classes.
//
// Each shape reaches named routes of the products on the Helium leg; the scalar leg (host, cortex-m55+nomve) runs
// every shape through its one row loop:
//   standard, K = 5, N = 13: the dual block (column 3), the single block (column 9), the remainder (column 12);
//   standard, K = 37, N = 5: the contiguous-rows kernel (column 1), the remainder (column 4);
//   packed, K = 5, N = 13, batch 5: the four-row block (rows 0-3) and the single row (row 4), a full column block
//   (column 2) and the partial one (column 12).
// Specials are rebuilt from bits held in volatiles and classified on bits, so a standalone -Ofast build of this file
// keeps them.

#define FC_NAN_MAX_BATCH 5
#define FC_NAN_MAX_K 37
#define FC_NAN_MAX_N 13
#define FC_NAN_BLOCK 4

typedef enum
{
    FC_NAN_INPUT = 0,
    FC_NAN_WEIGHT,
    FC_NAN_BIAS,
    FC_NAN_INF_TIMES_ZERO,
    FC_NAN_INF_MINUS_INF
} fc_nan_kind;

typedef enum
{
    FC_NAN_SAME = 0,
    FC_NAN_EXPECT_NAN,
    FC_NAN_EXPECT_MAX,
    FC_NAN_EXPECT_MIN
} fc_nan_expect;

typedef struct
{
    int32_t batch;
    int32_t k;
    int32_t n;
    int32_t packed;
    int32_t cols[2];
    int32_t n_cols;
    int32_t rows[2];
    int32_t n_rows;
} fc_nan_shape;

static const fc_nan_shape fc_nan_shapes[] = {
    {2, 5, 13, 0, {3, 9}, 2, {1, 0}, 1},
    {2, 5, 13, 0, {12, 0}, 1, {1, 0}, 1},
    {2, 37, 5, 0, {1, 4}, 2, {1, 0}, 1},
    {5, 5, 13, 1, {2, 12}, 2, {1, 4}, 2},
};

static float32_t fc_nan_x[FC_NAN_MAX_BATCH * FC_NAN_MAX_K];
static float32_t fc_nan_w[FC_NAN_MAX_N * FC_NAN_MAX_K];
static float32_t fc_nan_packed[((FC_NAN_MAX_N + FC_NAN_BLOCK - 1) / FC_NAN_BLOCK) * FC_NAN_BLOCK * FC_NAN_MAX_K];
static float32_t fc_nan_bias[FC_NAN_MAX_N];
static float32_t fc_nan_clean[FC_NAN_MAX_BATCH * FC_NAN_MAX_N];
static float32_t fc_nan_y[FC_NAN_MAX_BATCH * FC_NAN_MAX_N];
static fc_nan_expect fc_nan_want[FC_NAN_MAX_BATCH * FC_NAN_MAX_N];

static float32_t fc_nan_from_bits(volatile const uint32_t *bits)
{
    const uint32_t b = *bits;
    float32_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static uint32_t fc_nan_bits(float32_t x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int32_t fc_nan_is_nan(float32_t x) { return (fc_nan_bits(x) & 0x7FFFFFFFu) > 0x7F800000u; }

// Non-zero dyadic values of both signs, so every finite sum is exact and every weight has a sign.
static float32_t fc_nan_value(int32_t i, int32_t seed, float32_t scale)
{
    const uint32_t h = fc_hash(i, seed);
    const float32_t magnitude = (float32_t)((int32_t)(h & 7u) + 1) * scale;
    return (h & 8u) ? -magnitude : magnitude;
}

static void fc_nan_run(const fc_nan_shape *s, float32_t act_min, float32_t act_max, float32_t *out)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {s->batch, 1, 1, s->k};
    const cmsis_nn_dims filter_dims = {s->k, 1, 1, s->n};
    const cmsis_nn_dims bias_dims = {1, 1, 1, s->n};
    const cmsis_nn_dims output_dims = {s->batch, 1, 1, s->n};
    cmsis_nn_fc_params_f32 fc_params;

    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = act_min;
    fc_params.activation.max = act_max;
    fc_params.weight_format = s->packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;

    if (s->packed)
    {
        // [N / 4][K][4]: column c's tap k at ((c / 4) * K + k) * 4 + c % 4; the partial block's spare lanes are zero.
        memset(fc_nan_packed, 0, sizeof(fc_nan_packed));
        for (int32_t c = 0; c < s->n; ++c)
        {
            for (int32_t k = 0; k < s->k; ++k)
            {
                fc_nan_packed[((c / FC_NAN_BLOCK) * s->k + k) * FC_NAN_BLOCK + c % FC_NAN_BLOCK] =
                    fc_nan_w[c * s->k + k];
            }
        }
    }
    for (int32_t i = 0; i < s->batch * s->n; ++i)
    {
        out[i] = 0.0f;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f32(&ctx,
                                              &fc_params,
                                              &input_dims,
                                              fc_nan_x,
                                              &filter_dims,
                                              s->packed ? fc_nan_packed : fc_nan_w,
                                              &bias_dims,
                                              fc_nan_bias,
                                              &output_dims,
                                              out,
                                              ARM_NN_LAYOUT_NHWC));
}

static void fc_nan_case(fc_nan_kind kind, float32_t act_min, float32_t act_max)
{
    volatile uint32_t nan_bits = 0x7FC00000u;
    volatile uint32_t inf_bits = 0x7F800000u;
    volatile uint32_t ninf_bits = 0xFF800000u;
    const float32_t nan_v = fc_nan_from_bits(&nan_bits);
    const float32_t inf_v = fc_nan_from_bits(&inf_bits);
    const float32_t ninf_v = fc_nan_from_bits(&ninf_bits);

    for (size_t si = 0; si < sizeof(fc_nan_shapes) / sizeof(fc_nan_shapes[0]); ++si)
    {
        const fc_nan_shape *s = &fc_nan_shapes[si];
        const int32_t tap = s->k - 1;
        const int32_t seed = s->k * 7 + s->n;

        for (int32_t i = 0; i < s->batch * s->k; ++i)
        {
            fc_nan_x[i] = fc_nan_value(i, seed, 0.125f);
        }
        for (int32_t i = 0; i < s->n * s->k; ++i)
        {
            fc_nan_w[i] = fc_nan_value(i, seed + 1, 0.25f);
        }
        for (int32_t i = 0; i < s->n; ++i)
        {
            fc_nan_bias[i] = fc_nan_value(i, seed + 2, 0.125f);
        }
        for (int32_t i = 0; i < s->batch * s->n; ++i)
        {
            fc_nan_want[i] = FC_NAN_SAME;
        }

        // Finite set-up first, so the clean run already holds it.
        if (kind == FC_NAN_INF_TIMES_ZERO)
        {
            for (int32_t j = 0; j < s->n_cols; ++j)
            {
                fc_nan_w[s->cols[j] * s->k + tap] = 0.0f;
            }
        }
        if (kind == FC_NAN_INF_MINUS_INF)
        {
            // Features 0 and `tap` share a sign in every row, so +inf and -inf weights there give products of
            // opposite signs.
            for (int32_t r = 0; r < s->batch; ++r)
            {
                if ((fc_nan_x[r * s->k] < 0.0f) != (fc_nan_x[r * s->k + tap] < 0.0f))
                {
                    fc_nan_x[r * s->k + tap] = -fc_nan_x[r * s->k + tap];
                }
            }
        }
        fc_nan_run(s, act_min, act_max, fc_nan_clean);

        switch (kind)
        {
        case FC_NAN_INPUT:
            for (int32_t j = 0; j < s->n_rows; ++j)
            {
                fc_nan_x[s->rows[j] * s->k + tap] = nan_v;
                for (int32_t c = 0; c < s->n; ++c)
                {
                    fc_nan_want[s->rows[j] * s->n + c] = FC_NAN_EXPECT_NAN;
                }
            }
            break;
        case FC_NAN_WEIGHT:
        case FC_NAN_BIAS:
        case FC_NAN_INF_MINUS_INF:
            for (int32_t j = 0; j < s->n_cols; ++j)
            {
                const int32_t c = s->cols[j];
                if (kind == FC_NAN_WEIGHT)
                {
                    fc_nan_w[c * s->k + tap] = nan_v;
                }
                else if (kind == FC_NAN_BIAS)
                {
                    fc_nan_bias[c] = nan_v;
                }
                else
                {
                    // This column's dot meets an infinity of each sign: inf - inf.
                    fc_nan_w[c * s->k + 0] = inf_v;
                    fc_nan_w[c * s->k + tap] = ninf_v;
                }
                for (int32_t r = 0; r < s->batch; ++r)
                {
                    fc_nan_want[r * s->n + c] = FC_NAN_EXPECT_NAN;
                }
            }
            break;
        case FC_NAN_INF_TIMES_ZERO:
            // +inf meets a zero weight in the target columns (NaN) and a non-zero one elsewhere (+-inf, saturated).
            for (int32_t j = 0; j < s->n_rows; ++j)
            {
                const int32_t r = s->rows[j];
                fc_nan_x[r * s->k + tap] = inf_v;
                for (int32_t c = 0; c < s->n; ++c)
                {
                    const float32_t w = fc_nan_w[c * s->k + tap];
                    fc_nan_want[r * s->n + c] =
                        (w == 0.0f) ? FC_NAN_EXPECT_NAN : ((w > 0.0f) ? FC_NAN_EXPECT_MAX : FC_NAN_EXPECT_MIN);
                }
            }
            break;
        }

        fc_nan_run(s, act_min, act_max, fc_nan_y);

        for (int32_t i = 0; i < s->batch * s->n; ++i)
        {
            const uint32_t got = fc_nan_bits(fc_nan_y[i]);
            char msg[96];
            snprintf(msg,
                     sizeof(msg),
                     "shape %d output %d (row %d, column %d)",
                     (int)si,
                     (int)i,
                     (int)(i / s->n),
                     (int)(i % s->n));
            switch (fc_nan_want[i])
            {
            case FC_NAN_EXPECT_NAN:
                TEST_ASSERT_TRUE_MESSAGE(fc_nan_is_nan(fc_nan_y[i]), msg);
                break;
            case FC_NAN_EXPECT_MAX:
                TEST_ASSERT_EQUAL_HEX32_MESSAGE(fc_nan_bits(act_max), got, msg);
                break;
            case FC_NAN_EXPECT_MIN:
                TEST_ASSERT_EQUAL_HEX32_MESSAGE(fc_nan_bits(act_min), got, msg);
                break;
            default:
                TEST_ASSERT_EQUAL_HEX32_MESSAGE(fc_nan_bits(fc_nan_clean[i]), got, msg);
                break;
            }
        }
    }
}

// Each kind under the identity bounds and under a RELU6 bound, whose zero lower bound a NaN must pass too.
static void fc_nan_kind_both_bounds(fc_nan_kind kind)
{
    fc_nan_case(kind, -FLT_MAX, FLT_MAX);
    fc_nan_case(kind, 0.0f, 6.0f);
}

void fully_connected_nan_input_f32(void) { fc_nan_kind_both_bounds(FC_NAN_INPUT); }

void fully_connected_nan_weight_f32(void) { fc_nan_kind_both_bounds(FC_NAN_WEIGHT); }

void fully_connected_nan_bias_f32(void) { fc_nan_kind_both_bounds(FC_NAN_BIAS); }

void fully_connected_inf_times_zero_f32(void) { fc_nan_kind_both_bounds(FC_NAN_INF_TIMES_ZERO); }

void fully_connected_inf_minus_inf_f32(void) { fc_nan_kind_both_bounds(FC_NAN_INF_MINUS_INF); }

// One row, one class, on every backend: helia-ml's ADR 0074 ("Exactness is two layers") holds class decisions exact
// across backends. The row of finding 1 of the K3 rubric (one NaN rule in the float matmul products): x = [+inf, 1]
// against class rows [0, 1] [1, 0] [-1, 0] [-1, 1], no bias. Class 0 is 0 * inf = NaN, class 1 +inf, classes 2 and 3
// -inf. Before the rule the Helium single block wrote -FLT_MAX for class 0 and answered class 1, the scalar row loop
// wrote +FLT_MAX and answered class 0. With the NaN passed through, arm_argmax_f32 skips it and every leg answers
// class 1. The three-class form (finding 2) took the scalar remainder on both legs and answered the NaN's class 0.
void fully_connected_nan_row_one_class_f32(void)
{
    volatile uint32_t inf_bits = 0x7F800000u;
    const float32_t inf_v = fc_nan_from_bits(&inf_bits);
    const float32_t x[2] = {inf_v, 1.0f};
    const float32_t w[8] = {0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 0.0f, -1.0f, 1.0f};
    const cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_fc_params_f32 fc_params;

    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = -FLT_MAX;
    fc_params.activation.max = FLT_MAX;
    fc_params.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;

    for (int32_t classes = 4; classes >= 3; --classes)
    {
        const cmsis_nn_dims input_dims = {1, 1, 1, 2};
        const cmsis_nn_dims filter_dims = {2, 1, 1, classes};
        const cmsis_nn_dims bias_dims = {1, 1, 1, classes};
        const cmsis_nn_dims output_dims = {1, 1, 1, classes};
        const cmsis_nn_dims score_dims = {1, 1, 1, classes};
        float32_t scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int32_t label = -1;

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_fully_connected_f32(&ctx,
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
        TEST_ASSERT_TRUE_MESSAGE(fc_nan_is_nan(scores[0]), "class 0 is 0 * inf");
        TEST_ASSERT_EQUAL_HEX32(fc_nan_bits(FLT_MAX), fc_nan_bits(scores[1]));
        TEST_ASSERT_EQUAL_HEX32(fc_nan_bits(-FLT_MAX), fc_nan_bits(scores[2]));
        if (classes == 4)
        {
            TEST_ASSERT_EQUAL_HEX32(fc_nan_bits(-FLT_MAX), fc_nan_bits(scores[3]));
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_argmax_f32(scores, &score_dims, 3, &label));
        TEST_ASSERT_EQUAL_INT32(1, label);
    }
}

// A consumer can tell which rule its heliaCORE applies.
void fully_connected_nan_propagation_macro_f32(void)
{
#if defined(ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION) && (ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION == 1)
    TEST_PASS();
#else
    TEST_FAIL_MESSAGE("ARM_NN_HAS_MAT_MULT_FLT_NAN_PROPAGATION is not defined to 1");
#endif
}
