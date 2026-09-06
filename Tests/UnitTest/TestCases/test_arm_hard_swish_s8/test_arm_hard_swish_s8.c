/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <string.h>

#include "../Utils/validate.h"

/* Per-test generated data */
#include "../TestData/hard_swish_basic/test_data.h"
#include "../TestData/hard_swish_sweep_edges/test_data.h"
#include "../TestData/hard_swish_plateau_edges/test_data.h"
#include "../TestData/hard_swish_dense_mixture/test_data.h"
#include "../TestData/hard_swish_adversarial_step/test_data.h"
#include "../TestData/hard_swish_batched_wide/test_data.h"

#define REPEAT_NUM (1)

/* Helper to generate a Unity test for a given dataset.
 * NAME_UP:  UPPER_SNAKE prefix for the #defines
 * name_lc:  lower_snake prefix for the input/output symbols
 *
 * Expects the following macros from the generated headers:
 *   NAME_UP_INPUT_OFFSET
 *   NAME_UP_OUTPUT_OFFSET
 *   NAME_UP_RELU_MULTIPLIER_FP
 *   NAME_UP_RELU_MULTIPLIER_EXP
 *   NAME_UP_OUTPUT_MULTIPLIER_FP
 *   NAME_UP_OUTPUT_MULTIPLIER_EXP
 *   NAME_UP_OUTPUT_MULTIPLIER
 *   NAME_UP_OUTPUT_SHIFT
 *   NAME_UP_RELU_Q3
 *   NAME_UP_RELU_Q6
 *   NAME_UP_OUTPUT_LEN
 * and arrays:
 *   name_lc_input_tensor
 *   name_lc_output
 */
#define GEN_HS_TEST(NAME_UP, name_lc)                                                          \
    void name_lc##_arm_hard_swish_s8(void)                                                     \
    {                                                                                          \
        const int32_t output_size = NAME_UP##_OUTPUT_LEN;                                      \
                                                                                               \
        const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;                             \
        const int8_t *input_data = name_lc##_input_tensor;                                     \
        int8_t output[NAME_UP##_OUTPUT_LEN] = {0};                                             \
                                                                                               \
        for (int r = 0; r < REPEAT_NUM; ++r)                                                   \
        {                                                                                      \
            const arm_cmsis_nn_status result =                                                 \
                arm_hard_swish_compat_s8(                                                      \
                    input_data,                                                                \
                    NAME_UP##_INPUT_OFFSET,                                                    \
                    NAME_UP##_OUTPUT_OFFSET,                                                   \
                    NAME_UP##_OUTPUT_MULTIPLIER_FP,                                            \
                    NAME_UP##_OUTPUT_MULTIPLIER_EXP,                                           \
                    NAME_UP##_RELU_MULTIPLIER_FP,                                              \
                    NAME_UP##_RELU_MULTIPLIER_EXP,                                             \
                    output,                                                                    \
                    output_size                                                                \
                );                                                                             \
            TEST_ASSERT_EQUAL(expected, result);                                               \
        }                                                                                      \
                                                                                               \
        /* Allow no delta to be compat with TFLM */                                            \
        TEST_ASSERT_TRUE(validate_tol_s8(output, name_lc##_output, output_size, 0));           \
                                                                                               \
        for (int r = 0; r < REPEAT_NUM; ++r)                                                   \
        {                                                                                      \
            const arm_cmsis_nn_status result =                                                 \
                arm_hard_swish_precise_s8(                                                     \
                    input_data,                                                                \
                    NAME_UP##_INPUT_OFFSET,                                                    \
                    NAME_UP##_OUTPUT_OFFSET,                                                   \
                    NAME_UP##_OUTPUT_MULTIPLIER,                                               \
                    NAME_UP##_OUTPUT_SHIFT,                                                    \
                    NAME_UP##_RELU_Q3,                                                         \
                    NAME_UP##_RELU_Q6,                                                         \
                    0, /* prescale */                                                          \
                    output,                                                                    \
                    output_size                                                                \
                );                                                                             \
            TEST_ASSERT_EQUAL(expected, result);                                               \
        }                                                                                      \
                                                                                               \
        /* Allow small off-by-one since TFLM uses different rounding */                        \
        TEST_ASSERT_TRUE(validate_tol_s8(output, name_lc##_output, output_size, 1));           \
    }

/* Instantiate tests */
GEN_HS_TEST(HARD_SWISH_BASIC,            hard_swish_basic)
GEN_HS_TEST(HARD_SWISH_SWEEP_EDGES,      hard_swish_sweep_edges)
GEN_HS_TEST(HARD_SWISH_PLATEAU_EDGES,    hard_swish_plateau_edges)
GEN_HS_TEST(HARD_SWISH_DENSE_MIXTURE,    hard_swish_dense_mixture)
GEN_HS_TEST(HARD_SWISH_ADVERSARIAL_STEP, hard_swish_adversarial_step)
GEN_HS_TEST(HARD_SWISH_BATCHED_WIDE,     hard_swish_batched_wide)

/* ---------------------------------------------------------------------------------------------
 * #289: sizes around the 16-lane block and the full 256-value input ramp, goldens from the
 * scalar formula written out here (plain C, no library helpers), compared with zero tolerance.
 * Parameter sets: U = checked-in Unity data above, X = nsx-executorch lowering (prescale 0,
 * s_in = 16/255, s_out = 8.375/255), A = helia-aot precise lowering (s = 1/128, prescale 18),
 * T = helia-core-tester hard_swish_compat descriptors (extras 1/128; TFLite of uniform[-8,8]).
 * ------------------------------------------------------------------------------------------ */

static int32_t hs_ref_clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

static int32_t hs_ref_div_pot(int32_t x, int32_t e)
{
    // rounding divide by 2^e, midpoint away from zero (arm_nn_divide_by_power_of_two semantics)
    if (e == 0)
    {
        return x;
    }
    const int32_t mask = (int32_t)((1u << e) - 1u);
    const int32_t rem = x & mask;
    int32_t res = x >> e;
    int32_t thr = mask >> 1;
    if (res < 0)
    {
        thr += 1;
    }
    if (rem > thr)
    {
        res += 1;
    }
    return res;
}

static int8_t hs_ref_precise(int8_t in,
                             int32_t zi,
                             int32_t zo,
                             int32_t mult,
                             int32_t shift,
                             int32_t q3,
                             int32_t q6,
                             int32_t prescale)
{
    const int32_t x = (int32_t)in - zi;
    int32_t xr = hs_ref_clamp(x + q3, 0, q6);
    if (prescale > 0)
    {
        xr = (int32_t)(((uint32_t)xr + (1u << (prescale - 1))) >> prescale);
    }
    const int32_t left = shift > 0 ? shift : 0;
    const int32_t right = shift < 0 ? -shift : 0;
    const int64_t prod = (int64_t)(int32_t)((uint32_t)(x * xr) << left) * (int64_t)mult;
    const int32_t high = (int32_t)((prod + (1LL << 30)) >> 31);
    int32_t y = hs_ref_div_pot(high, right) + zo;
    return (int8_t)hs_ref_clamp(y, INT8_MIN, INT8_MAX);
}

static int16_t hs_ref_sat16(int32_t v) { return (int16_t)hs_ref_clamp(v, INT16_MIN, INT16_MAX); }

static int16_t hs_ref_sqrdmulh16(int16_t a, int16_t b)
{
    if (a == INT16_MIN && b == INT16_MIN)
    {
        return INT16_MAX;
    }
    return hs_ref_sat16((((int32_t)a * b) * 2 + (1 << 15)) >> 16);
}

static int8_t hs_ref_compat(int8_t in, int32_t zi, int32_t zo, int32_t ofp, int32_t oexp, int32_t rfp, int32_t rexp)
{
    const int16_t x = (int16_t)((int32_t)in - zi);
    const int16_t hires = hs_ref_sat16((int32_t)x << 7);
    const int16_t y_pre = hs_ref_sqrdmulh16(hires, (int16_t)ofp);
    int16_t rel = hires;
    if (rexp > 0)
    {
        rel = (rexp - 1 > 0) ? hs_ref_sat16((int32_t)rel << (rexp - 1)) : rel;
        rel = hs_ref_sqrdmulh16(rel, (int16_t)rfp);
        rel = hs_ref_sat16((int32_t)rel << 1);
    }
    else if (rexp < 0)
    {
        rel = hs_ref_sqrdmulh16(rel, (int16_t)rfp);
        rel = hs_ref_sat16(hs_ref_div_pot(rel, -rexp));
    }
    else
    {
        rel = hs_ref_sqrdmulh16(rel, (int16_t)rfp);
    }
    rel = (int16_t)(((int32_t)rel + 32768) >> 1);
    // non-rounded doubling high multiply: truncation toward zero of (a*b)/2^15
    int32_t y = ((int32_t)rel * y_pre) / (1 << 15);
    if (rel == INT16_MIN && y_pre == INT16_MIN)
    {
        y = INT16_MAX;
    }
    y = hs_ref_sat16(y);
    if (oexp < 0)
    {
        y = hs_ref_sat16(hs_ref_div_pot(y, -oexp));
    }
    return (int8_t)hs_ref_clamp(y + zo, INT8_MIN, INT8_MAX);
}

#define HS289_MAX_SIZE 8191
#define HS289_GUARD 32
// 1760 / 4160 are the MVE table thresholds of the precise / compat kernels (#289)
static const int32_t hs289_sizes[] = {1, 15, 16, 17, 33, 256, 1000, 1759, 1760, 1761, 4159, 4160, 4161, 8191};
#define HS289_N_SIZES (int)(sizeof(hs289_sizes) / sizeof(hs289_sizes[0]))

static void hs289_fill_input(int8_t *in, int32_t n, int ramp)
{
    for (int32_t i = 0; i < n; i++)
    {
        in[i] = ramp ? (int8_t)(i - 128) : (int8_t)((i * 37 + 11) & 0xff);
    }
}

typedef struct
{
    int32_t zi, zo, mult, shift, q3, q6, prescale;
} hs289_precise_params;

typedef struct
{
    int32_t zi, zo, ofp, oexp, rfp, rexp;
} hs289_compat_params;

static const hs289_precise_params hs289_precise_sets[] = {
    {-128, -128, 2041101380, -10, 822, 1645, 0}, // U
    {0, -117, 1372914724, -5, 48, 96, 0},        // X
    {0, 0, 1431655765, 9, 384, 768, 18},         // A
};
static const hs289_compat_params hs289_compat_sets[] = {
    {-128, -128, 25012, -6, 20401, -1}, // U
    {0, 0, 16384, -6, 21845, 0},        // T exp0
    {0, -117, 31301, -6, 21931, 3},     // T uniform[-8,8]
};

static void hs289_run_precise(const hs289_precise_params *p, int32_t n, int ramp)
{
    static int8_t in[HS289_MAX_SIZE], expected[HS289_MAX_SIZE], out[HS289_MAX_SIZE + HS289_GUARD];
    hs289_fill_input(in, n, ramp);
    for (int32_t i = 0; i < n; i++)
    {
        expected[i] = hs_ref_precise(in[i], p->zi, p->zo, p->mult, p->shift, p->q3, p->q6, p->prescale);
    }
    memset(out, 0x5a, sizeof(out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_hard_swish_precise_s8(in, p->zi, p->zo, p->mult, p->shift, p->q3, p->q6, p->prescale, out, n));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, n);
    for (int32_t i = n; i < n + HS289_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5a, out[i]);
    }
}

static void hs289_run_compat(const hs289_compat_params *p, int32_t n, int ramp)
{
    static int8_t in[HS289_MAX_SIZE], expected[HS289_MAX_SIZE], out[HS289_MAX_SIZE + HS289_GUARD];
    hs289_fill_input(in, n, ramp);
    for (int32_t i = 0; i < n; i++)
    {
        expected[i] = hs_ref_compat(in[i], p->zi, p->zo, p->ofp, p->oexp, p->rfp, p->rexp);
    }
    memset(out, 0x5a, sizeof(out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_hard_swish_compat_s8(in, p->zi, p->zo, p->ofp, p->oexp, p->rfp, p->rexp, out, n));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, n);
    for (int32_t i = n; i < n + HS289_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5a, out[i]);
    }
}

void hard_swish_precise_sizes_arm_hard_swish_s8(int set)
{
    for (int s = 0; s < HS289_N_SIZES; s++)
    {
        hs289_run_precise(&hs289_precise_sets[set], hs289_sizes[s], 0);
    }
}

void hard_swish_compat_sizes_arm_hard_swish_s8(int set)
{
    for (int s = 0; s < HS289_N_SIZES; s++)
    {
        hs289_run_compat(&hs289_compat_sets[set], hs289_sizes[s], 0);
    }
}

void hard_swish_precise_ramp256_arm_hard_swish_s8(void)
{
    for (int set = 0; set < 3; set++)
    {
        hs289_run_precise(&hs289_precise_sets[set], 256, 1);
    }
}

void hard_swish_compat_ramp256_arm_hard_swish_s8(void)
{
    for (int set = 0; set < 3; set++)
    {
        hs289_run_compat(&hs289_compat_sets[set], 256, 1);
    }
}
