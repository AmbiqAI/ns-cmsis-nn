/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>
#include "../Utils/mpu_guard.h"

typedef arm_cmsis_nn_status (*gelu_function)(const float32_t *, float32_t *, int32_t);

static uint32_t gelu_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static float gelu_float(uint32_t u)
{
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

/* C adaptation of GeluTransform in TensorFlow v2.16.1 reference/gelu.h.
 * Keep the decimal producer constant independent of the kernel's encoding.
 * The separately compiled upstream C++ producer is the author parity control.
 */
static float gelu_producer(float x)
{
    /* Force the linked libm call, not compiler-folded transcendental results. */
    float (*volatile runtime_erfc)(float) = erfcf;
    return 0.5f * x * runtime_erfc(x * (float)-0.70710678118654752440);
}

static bool gelu_equal(float a, float b)
{
    uint32_t ua = gelu_bits(a), ub = gelu_bits(b);
    if ((ua & 0x7fffffffU) > 0x7f800000U)
    {
        return (ub & 0x7fffffffU) > 0x7f800000U;
    }
    return ua == ub;
}

static bool gelu_finite_control(gelu_function fn)
{
    float input[67], output[67];
    for (int i = 0; i < 67; ++i)
    {
        input[i] = (float)(i - 43) / 8.0f;
        output[i] = 99.0f;
    }
    /* Compare scalar calls: optimized array calls may use a different vector libm. */
    for (int i = 0; i < 67; ++i)
    {
        if (fn(input + i, output + i, 1) != ARM_CMSIS_NN_SUCCESS ||
            !gelu_equal(gelu_producer(input[i]), output[i]))
        {
            return false;
        }
    }
    /* A wrong ReLU-like replacement must not erase the negative lobe. */
    return output[35] < -0.1f && output[35] > -0.2f && output[3] < 0.0f;
}

static bool gelu_count_control(gelu_function fn)
{
    const int counts[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33};
    float input[36], output[36];
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c)
    {
        for (int i = 0; i < 36; ++i)
        {
            /* Here erfc rounds to 2, so GELU is exactly the input on either libm path. */
            input[i] = 32.0f + (float)i;
            output[i] = 99.0f;
        }
        int n = counts[c];
        /* Four-byte offset: no vector alignment may be required. */
        if (fn(input + 1, output + 1, n) != ARM_CMSIS_NN_SUCCESS || output[0] != 99.0f ||
            output[n + 1] != 99.0f)
        {
            return false;
        }
        for (int i = 1; i <= n; ++i)
        {
            if (gelu_bits(input[i]) != gelu_bits(output[i]))
            {
                return false;
            }
        }
    }
    return true;
}

void gelu_f32_finite(void)
{
    TEST_ASSERT_TRUE(gelu_finite_control(arm_nn_gelu_f32));
}

void gelu_f32_counts(void)
{
    TEST_ASSERT_TRUE(gelu_count_control(arm_nn_gelu_f32));
}

void gelu_f32_in_place(void)
{
    float values[17], reference[17];
    /* Array aliasing uses exact identity inputs, independent of libm vectorization. */
    for (int i = 0; i < 17; ++i)
    {
        values[i] = reference[i] = 32.0f + (float)i;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f32(values, values, 17));
    for (int i = 0; i < 17; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(gelu_bits(reference[i]), gelu_bits(values[i]));
    }
    /* Nontrivial in-place arithmetic retains the controlled scalar producer gate. */
    for (int i = 0; i < 17; ++i)
    {
        float value = (float)(i - 12) / 4.0f;
        const float expected = gelu_producer(value);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f32(&value, &value, 1));
        TEST_ASSERT_TRUE(gelu_equal(expected, value));
    }
}

void gelu_f32_special(void)
{
    const uint32_t patterns[] = {0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                                 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
                                 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345};
    for (unsigned i = 0; i < sizeof(patterns) / sizeof(patterns[0]); ++i)
    {
        float input = gelu_float(patterns[i]), output;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f32(&input, &output, 1));
        TEST_ASSERT_TRUE(gelu_equal(gelu_producer(input), output));
    }
}

void gelu_f32_arguments(void)
{
    float x = 1.0f, y = 99.0f;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f32(NULL, NULL, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f32(&x, &y, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f32(&x, &y, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f32(NULL, &y, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f32(&x, NULL, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f32(NULL, NULL, INT32_MIN));
    TEST_ASSERT_EQUAL_HEX32(0x42c60000, gelu_bits(y));
}

void gelu_f32_bounds(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    float input[17], output[17], expected[17];
    const int counts[] = {1, 3, 4, 5, 17};
    for (int i = 0; i < 17; ++i)
    {
        input[i] = expected[i] = 32.0f + (float)i;
    }
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c)
    {
        const int n = counts[c];
        const float *guarded_input = guard_place(input, n * sizeof(float));
        guard_gap_enable();
        arm_cmsis_nn_status status = arm_nn_gelu_f32(guarded_input, output, n);
        guard_gap_disable();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
        for (int i = 0; i < n; ++i)
        {
            TEST_ASSERT_TRUE(gelu_equal(expected[i], output[i]));
        }
        float *guarded_output = guard_end(n * sizeof(float));
        guard_gap_enable();
        status = arm_nn_gelu_f32(input, guarded_output, n);
        guard_gap_disable();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
        for (int i = 0; i < n; ++i)
        {
            TEST_ASSERT_TRUE(gelu_equal(expected[i], guarded_output[i]));
        }
    }
#else
    TEST_IGNORE_MESSAGE("MPU guard requires Corstone-300/MVE");
#endif
}
