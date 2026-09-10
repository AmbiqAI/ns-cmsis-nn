/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Host -Ofast NaN probe for the six float elementwise clamp lanes (#334 item d).
 *
 * The Unity harness deliberately builds every test TU with -fno-finite-math-only
 * (Tests/UnitTest/CMakeLists.txt), so the harness suites cannot observe what the
 * shipped -Ofast does to the NaN contract. This probe exists precisely to run
 * under the hostile flags: its CMakeLists compiles this TU AND its own copies of
 * the six elementwise kernels at -Ofast -ffinite-math-only, the flag pair #333
 * showed folds a floating-point NaN test away. The clamp restored by #380
 * classifies NaN on the integer bit pattern, over which -ffinite-math-only
 * grants the compiler no license, so every lane below must still return NaN.
 *
 * Inputs are staged through volatile bit patterns so the compiler cannot
 * constant-fold the non-finite arithmetic away, and the NaN check is a
 * bit-pattern classification for the same reason isnan() is not used: under
 * -ffinite-math-only isnan() may fold to a constant false.
 *
 * Exit status is the number of failed lanes; any clamp-to-bound on an
 * expected-NaN element is a failure. Wired into the host-sanitizer ctest run.
 */

#include <arm_nnfunctions.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float32_t f32_from_bits(volatile const uint32_t *bits)
{
    const uint32_t b = *bits;
    float32_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static int f32_bits_are_nan(float32_t x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

static float16_t f16_from_bits(volatile const uint16_t *bits)
{
    const uint16_t b = *bits;
    float16_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static int f16_bits_are_nan(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (uint16_t)(bits & 0x7FFFu) > 0x7C00u;
}

/* Report one lane: 0 on pass, 1 on failure. `nan_count` leading output elements
 * must be NaN; the two elements after them must be the clamped bounds +6/-6. */
static int check_f32_lane(const char *lane, arm_cmsis_nn_status status, const float32_t *out, int nan_count)
{
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        fprintf(stderr, "FAIL %s: kernel returned %d\n", lane, (int)status);
        return 1;
    }
    for (int i = 0; i < nan_count; ++i)
    {
        if (!f32_bits_are_nan(out[i]))
        {
            fprintf(stderr, "FAIL %s: element %d clamped to %f instead of NaN\n", lane, i, (double)out[i]);
            return 1;
        }
    }
    if (out[nan_count] != 6.0f || out[nan_count + 1] != -6.0f)
    {
        fprintf(stderr, "FAIL %s: finite elements did not clamp to the bounds\n", lane);
        return 1;
    }
    return 0;
}

static int check_f16_lane(const char *lane, arm_cmsis_nn_status status, const float16_t *out, int nan_count)
{
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        fprintf(stderr, "FAIL %s: kernel returned %d\n", lane, (int)status);
        return 1;
    }
    for (int i = 0; i < nan_count; ++i)
    {
        if (!f16_bits_are_nan(out[i]))
        {
            fprintf(stderr, "FAIL %s: element %d clamped to %f instead of NaN\n", lane, i, (double)(float32_t)out[i]);
            return 1;
        }
    }
    if ((float32_t)out[nan_count] != 6.0f || (float32_t)out[nan_count + 1] != -6.0f)
    {
        fprintf(stderr, "FAIL %s: finite elements did not clamp to the bounds\n", lane);
        return 1;
    }
    return 0;
}

int main(void)
{
    int failures = 0;

    volatile uint32_t inf32_bits = 0x7F800000u;
    volatile uint32_t nan32_bits = 0x7FC00000u;
    const float32_t inf32 = f32_from_bits(&inf32_bits);
    const float32_t nan32 = f32_from_bits(&nan32_bits);

    volatile uint32_t neg_inf32_bits = 0xFF800000u;
    const float32_t neg_inf32 = f32_from_bits(&neg_inf32_bits);

    volatile uint16_t inf16_bits = 0x7C00u;
    volatile uint16_t nan16_bits = 0x7E00u;
    const float16_t inf16 = f16_from_bits(&inf16_bits);
    const float16_t nan16 = f16_from_bits(&nan16_bits);

    volatile uint16_t neg_inf16_bits = 0xFC00u;
    const float16_t neg_inf16 = f16_from_bits(&neg_inf16_bits);

    {
        /* NaN operand, NaN from Inf + (-Inf), then both clamp bounds. */
        const float32_t in1[4] = {nan32, inf32, 8.0f, -8.0f};
        const float32_t in2[4] = {0.0f, neg_inf32, 1.0f, -1.0f};
        float32_t out[4] = {0};
        failures += check_f32_lane("add_f32", arm_elementwise_add_f32(in1, in2, out, -6.0f, 6.0f, 4), out, 2);
    }
    {
        /* NaN operand, NaN from Inf - Inf, then both clamp bounds. */
        const float32_t in1[4] = {nan32, inf32, 8.0f, -8.0f};
        const float32_t in2[4] = {0.0f, inf32, 1.0f, -1.0f};
        float32_t out[4] = {0};
        failures += check_f32_lane("sub_f32", arm_elementwise_sub_f32(in1, in2, out, -6.0f, 6.0f, 4), out, 2);
    }
    {
        /* NaN operand, NaN from 0 * Inf, then both clamp bounds. */
        const float32_t in1[4] = {nan32, 0.0f, 8.0f, -8.0f};
        const float32_t in2[4] = {1.0f, inf32, 1.0f, 1.0f};
        float32_t out[4] = {0};
        failures += check_f32_lane("mul_f32", arm_elementwise_mul_f32(in1, in2, out, -6.0f, 6.0f, 4), out, 2);
    }

    {
        const float16_t in1[4] = {nan16, inf16, (float16_t)8.0f, (float16_t)-8.0f};
        const float16_t in2[4] = {(float16_t)0.0f, neg_inf16, (float16_t)1.0f, (float16_t)-1.0f};
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane(
            "add_f16", arm_elementwise_add_f16(in1, in2, out, (float16_t)-6.0f, (float16_t)6.0f, 4), out, 2);
    }
    {
        const float16_t in1[4] = {nan16, inf16, (float16_t)8.0f, (float16_t)-8.0f};
        const float16_t in2[4] = {(float16_t)0.0f, inf16, (float16_t)1.0f, (float16_t)-1.0f};
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane(
            "sub_f16", arm_elementwise_sub_f16(in1, in2, out, (float16_t)-6.0f, (float16_t)6.0f, 4), out, 2);
    }
    {
        const float16_t in1[4] = {nan16, (float16_t)0.0f, (float16_t)8.0f, (float16_t)-8.0f};
        const float16_t in2[4] = {(float16_t)1.0f, inf16, (float16_t)1.0f, (float16_t)1.0f};
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane(
            "mul_f16", arm_elementwise_mul_f16(in1, in2, out, (float16_t)-6.0f, (float16_t)6.0f, 4), out, 2);
    }

    if (failures == 0)
    {
        printf("elementwise -Ofast NaN probe: 6/6 lanes propagate\n");
    }
    return failures;
}
