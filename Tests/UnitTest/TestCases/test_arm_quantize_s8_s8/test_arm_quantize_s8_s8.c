/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

 #include "arm_nnfunctions.h"
 #include "unity.h"

#include "../Utils/validate.h"


void test_arm_quantize_s8_s8(void)
{
    const arm_cmsis_nn_status expected_status = ARM_CMSIS_NN_SUCCESS;
    // Prepare test data
    const int8_t input[] = {-128, -64, 0, 63, 64, 127};
    enum { INPUT_LEN = (int)(sizeof(input) / sizeof(input[0])) };

    // Parameters for identity
    const int32_t effective_scale_multiplier = 0x40000000; // Q1.31 format close to 1.0
    const int32_t effective_scale_shift = 0;
    const int32_t input_zeropoint = 0;
    const int32_t output_zeropoint = 0;

    int8_t output[INPUT_LEN];
    int8_t expected[INPUT_LEN];

    // Compute the "reference" expected results in float
    for (size_t i = 0; i < INPUT_LEN; i++)
    {
        float ref_val = 0.5f * input[i]; // scale by 0.5
        // Round to nearest integer, then clamp to [-128, 127]
        int32_t rounded = (int32_t)((ref_val >= 0) ? (ref_val + 0.5f) : (ref_val - 0.5f));
        if (rounded > 127)
        {
            rounded = 127;
        }
        else if (rounded < -128)
        {
            rounded = -128;
        }
        expected[i] = (int8_t)rounded;
    }

    // Call the function under test
    arm_cmsis_nn_status result = arm_requantize_s8_s8(
        input,
        output,
        INPUT_LEN,
        effective_scale_multiplier,
        effective_scale_shift,
        input_zeropoint,
        output_zeropoint
    );

    // Verify the results
    TEST_ASSERT_EQUAL(expected_status, result);
    for (size_t i = 0; i < INPUT_LEN; i++)
    {
        // For identity transform, output[i] should match input[i]
        TEST_ASSERT_EQUAL_INT8_MESSAGE(expected[i], output[i], "Identity requantization failed");
    }
}

/* Reference requantization in int64, written independently of arm_nn_requantize(): a multiply, rounding half up at
 * bit 31, then a divide by 2^-shift rounding half away from zero; or, with CMSIS_NN_USE_SINGLE_ROUNDING, one divide by
 * 2^(31 - shift) rounding half up. */
static int64_t requant_floor_div_pow2(const int64_t x, const int32_t e)
{
    const int64_t d = (int64_t)1 << e;
    return x >= 0 ? x / d : -((-x + d - 1) / d);
}

static int64_t requant_ref(const int32_t x, const int32_t mult, const int32_t shift)
{
    if (shift > 0)
    {
        return requant_floor_div_pow2((int64_t)x * mult + ((int64_t)1 << (30 - shift)), 31 - shift);
    }
#if defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    const int32_t total_shift = 31 - shift;
    return requant_floor_div_pow2((int64_t)x * mult + ((int64_t)1 << (total_shift - 1)), total_shift);
#else
    const int32_t left = shift > 0 ? shift : 0;
    const int32_t right = shift > 0 ? 0 : -shift;
    const int64_t high = requant_floor_div_pow2((int64_t)x * ((int64_t)1 << left) * mult + ((int64_t)1 << 30), 31);
    if (right == 0)
    {
        return (int32_t)high;
    }
    const int64_t half = (int64_t)1 << (right - 1);
    return (int32_t)(high >= 0 ? requant_floor_div_pow2(high + half, right) : -requant_floor_div_pow2(-high + half, right));
#endif
}

typedef enum
{
    REQUANT_S8_S8,
    REQUANT_S8_U8,
    REQUANT_U8_S8
} requant_kind_t;

typedef struct
{
    int32_t mult;
    int32_t shift;
} requant_scale_t;

/* Ordinary scales, the vector/wide boundary, and legal extreme shifts with saturating and unsaturated results. */
static const requant_scale_t requant_scales[] = {
    {1 << 30, 1}, {1 << 30, 0}, {1610612736, 0}, {1771674010, 2}, {1690522173, -6}, {1 << 30, -20},
    {INT32_MAX, 22}, {1 << 30, 23}, {1 << 30, 24}, {INT32_MAX, 30}, {1073741823, 30},
    {1, 23}, {1, 30}, {0, 30}, {INT32_MAX, -31}};

#define REQUANT_CANARY 16

static void requant_run(const requant_kind_t kind,
                        const void *in,
                        void *out,
                        const int32_t size,
                        const requant_scale_t s,
                        const int32_t in_zp,
                        const int32_t out_zp)
{
    arm_cmsis_nn_status status;
    switch (kind)
    {
    case REQUANT_S8_S8:
        status = arm_requantize_s8_s8((const int8_t *)in, (int8_t *)out, size, s.mult, s.shift, in_zp, out_zp);
        break;
    case REQUANT_S8_U8:
        status = arm_requantize_s8_u8((const int8_t *)in, (uint8_t *)out, size, s.mult, s.shift, in_zp, out_zp);
        break;
    default:
        status = arm_requantize_u8_s8((const uint8_t *)in, (int8_t *)out, size, s.mult, s.shift, in_zp, out_zp);
        break;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
}

/* Every input byte, every scale and a spread of zero points, against requant_ref(); lengths 1 to 19 at several
 * offsets cover the Helium tail, and the bytes after each output must stay untouched. */
static void requant_check(const requant_kind_t kind)
{
    const bool in_unsigned = kind == REQUANT_U8_S8;
    const bool out_unsigned = kind == REQUANT_S8_U8;
    const int32_t in_zps_s8[] = {-128, -5, 0, 127};
    const int32_t in_zps_u8[] = {0, 3, 128, 255};
    const int32_t out_zps_s8[] = {-128, 0, 9, 127};
    const int32_t out_zps_u8[] = {0, 1, 128, 255};
    const int32_t *in_zps = in_unsigned ? in_zps_u8 : in_zps_s8;
    const int32_t *out_zps = out_unsigned ? out_zps_u8 : out_zps_s8;
    const int32_t out_min = out_unsigned ? 0 : -128;
    const int32_t out_max = out_unsigned ? 255 : 127;

    uint8_t in[256];
    uint8_t out[256 + REQUANT_CANARY];
    for (int32_t i = 0; i < 256; i++)
    {
        in[i] = (uint8_t)i;
    }

    for (size_t si = 0; si < sizeof(requant_scales) / sizeof(requant_scales[0]); si++)
    {
        for (int32_t zi = 0; zi < 4; zi++)
        {
            for (int32_t zo = 0; zo < 4; zo++)
            {
                int8_t expected[256];
                for (int32_t i = 0; i < 256; i++)
                {
                    const int32_t x = in_unsigned ? (int32_t)in[i] : (int32_t)(int8_t)in[i];
                    int64_t y = requant_ref(x - in_zps[zi], requant_scales[si].mult, requant_scales[si].shift);
                    y += out_zps[zo];
                    y = y < out_min ? out_min : (y > out_max ? out_max : y);
                    expected[i] = (int8_t)(uint8_t)y;
                }

                memset(out, 0x5A, sizeof(out));
                requant_run(kind, in, out, 256, requant_scales[si], in_zps[zi], out_zps[zo]);
                TEST_ASSERT_EQUAL_INT8_ARRAY(expected, (const int8_t *)out, 256);
                for (int32_t i = 256; i < 256 + REQUANT_CANARY; i++)
                {
                    TEST_ASSERT_EQUAL_HEX8(0x5A, out[i]);
                }

                for (int32_t offset = 0; offset < 256 - 32; offset += 61)
                {
                    for (int32_t len = 1; len < 20; len++)
                    {
                        memset(out, 0x5A, sizeof(out));
                        requant_run(kind, in + offset, out, len, requant_scales[si], in_zps[zi], out_zps[zo]);
                        TEST_ASSERT_EQUAL_INT8_ARRAY(expected + offset, (const int8_t *)out, len);
                        for (int32_t i = len; i < len + REQUANT_CANARY; i++)
                        {
                            TEST_ASSERT_EQUAL_HEX8(0x5A, out[i]);
                        }
                    }
                }
            }
        }
    }

    /* A length of 0 or below writes nothing */
    memset(out, 0x5A, sizeof(out));
    requant_run(kind, in, out, 0, requant_scales[0], 0, 0);
    requant_run(kind, in, out, -1, requant_scales[0], 0, 0);
    requant_run(kind, in, out, INT32_MIN, requant_scales[0], 0, 0);
    for (int32_t i = 0; i < REQUANT_CANARY; i++)
    {
        TEST_ASSERT_EQUAL_HEX8(0x5A, out[i]);
    }

    /* These objects end at the predicated tail, without the matrix's spare backing storage. */
    const uint8_t one_input = 255;
    uint8_t one_output = 0;
    const uint8_t last = in_unsigned ? 127 : (out_unsigned ? 0 : 255);
    requant_run(kind, &one_input, &one_output, 1, requant_scales[0], 0, 0);
    TEST_ASSERT_EQUAL_HEX8(last, one_output);
    const uint8_t three_input[3] = {0, 127, 255};
    uint8_t three_output[3] = {0};
    const uint8_t three_expected[3] = {0, 127, last};
    requant_run(kind, three_input, three_output, 3, requant_scales[0], 0, 0);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(three_expected, three_output, 3);
}

void test_arm_requantize_s8_s8_all_inputs(void) { requant_check(REQUANT_S8_S8); }

void test_arm_requantize_s8_u8_all_inputs(void) { requant_check(REQUANT_S8_U8); }

void test_arm_requantize_u8_s8_all_inputs(void) { requant_check(REQUANT_U8_S8); }

/* microWakeWord's output QUANTIZE: int8 at scale 1/256, zero point -128, to uint8 at the same scale, zero point 0,
 * is the byte plus 128 */
void test_arm_requantize_s8_u8_microwakeword(void)
{
    int8_t in[256];
    uint8_t out[256];
    for (int32_t i = 0; i < 256; i++)
    {
        in[i] = (int8_t)(i - 128);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8_u8(in, out, 256, 1 << 30, 1, -128, 0));
    for (int32_t i = 0; i < 256; i++)
    {
        TEST_ASSERT_EQUAL_UINT8((uint8_t)i, out[i]);
    }
}
