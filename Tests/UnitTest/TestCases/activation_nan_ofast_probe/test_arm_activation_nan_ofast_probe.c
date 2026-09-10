/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Host -Ofast NaN probe for the float activation legs (#382).
 *
 * The Unity harness deliberately builds every test TU with -fno-finite-math-only
 * (Tests/UnitTest/CMakeLists.txt), so the harness suites cannot observe what the
 * shipped -Ofast does to the NaN contract. This probe exists precisely to run
 * under the hostile flags: its CMakeLists compiles this TU AND its own copies of
 * arm_nn_activation_f32/f16 at -Ofast -ffinite-math-only, the flag pair #333
 * showed folds a floating-point NaN test away. Before #382 the f32 RELU, RELU6
 * and LEAKY_RELU legs decided NaN with plain floating-point ternaries, whose
 * result under that pair is a property of the codegen rather than a contract;
 * they now restore the NaN in the integer domain, over which -ffinite-math-only
 * grants the compiler no license, so every lane below must return NaN.
 *
 * Scope: on an x86 host neither ARM_MATH_MVEF nor ARM_MATH_MVE_FLOAT16 is
 * defined, so only the SCALAR legs run here. The MVE RELU/RELU6 legs that #382
 * also changed cannot execute on this host and are covered on target instead.
 *
 * Inputs are staged through volatile bit patterns so the compiler cannot
 * constant-fold the non-finite arithmetic away, and the NaN check is a
 * bit-pattern classification for the same reason isnan() is not used: under
 * -ffinite-math-only isnan() may fold to a constant false.
 *
 * Each lane also asserts finite neutrality, so a fix that propagated NaN by
 * breaking the ordinary values would still fail. Exit status is the number of
 * failed lanes. Wired into the host-sanitizer ctest run.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
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

/* Every lane drives the same four-element input: NaN, then three finite probes
 * chosen so the expected outputs differ per activation. Element 0 must come
 * back NaN; elements 1..3 must match the caller's finite expectation exactly. */
static int check_f32_lane(const char *lane,
                          arm_cmsis_nn_status status,
                          const float32_t *out,
                          const float32_t *expect_finite)
{
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        fprintf(stderr, "FAIL %s: kernel returned %d\n", lane, (int)status);
        return 1;
    }
    if (!f32_bits_are_nan(out[0]))
    {
        fprintf(stderr, "FAIL %s: NaN input came back as %f\n", lane, (double)out[0]);
        return 1;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (out[i + 1] != expect_finite[i])
        {
            fprintf(stderr,
                    "FAIL %s: finite element %d is %f, expected %f\n",
                    lane,
                    i + 1,
                    (double)out[i + 1],
                    (double)expect_finite[i]);
            return 1;
        }
    }
    return 0;
}

static int check_f16_lane(const char *lane,
                          arm_cmsis_nn_status status,
                          const float16_t *out,
                          const float32_t *expect_finite)
{
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        fprintf(stderr, "FAIL %s: kernel returned %d\n", lane, (int)status);
        return 1;
    }
    if (!f16_bits_are_nan(out[0]))
    {
        fprintf(stderr, "FAIL %s: NaN input came back as %f\n", lane, (double)(float32_t)out[0]);
        return 1;
    }
    for (int i = 0; i < 3; ++i)
    {
        if ((float32_t)out[i + 1] != expect_finite[i])
        {
            fprintf(stderr,
                    "FAIL %s: finite element %d is %f, expected %f\n",
                    lane,
                    i + 1,
                    (double)(float32_t)out[i + 1],
                    (double)expect_finite[i]);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    int failures = 0;

    volatile uint32_t nan32_bits = 0x7FC00000u;
    const float32_t nan32 = f32_from_bits(&nan32_bits);

    volatile uint16_t nan16_bits = 0x7E00u;
    const float16_t nan16 = f16_from_bits(&nan16_bits);

    /* -2 exercises the negative leg, 3 the identity leg, 8 the RELU6 upper bound. */
    const float32_t in32[4] = {nan32, -2.0f, 3.0f, 8.0f};
    const float16_t in16[4] = {nan16, (float16_t)-2.0f, (float16_t)3.0f, (float16_t)8.0f};
    const float32_t alpha = 0.5f;

    {
        const float32_t expect[3] = {0.0f, 3.0f, 8.0f};
        float32_t out[4] = {0};
        failures +=
            check_f32_lane("relu_f32", arm_nn_activation_f32(in32, out, 4, ARM_NN_FLT_ACT_RELU, 0.0f), out, expect);
    }
    {
        const float32_t expect[3] = {0.0f, 3.0f, 6.0f};
        float32_t out[4] = {0};
        failures +=
            check_f32_lane("relu6_f32", arm_nn_activation_f32(in32, out, 4, ARM_NN_FLT_ACT_RELU6, 0.0f), out, expect);
    }
    {
        const float32_t expect[3] = {-1.0f, 3.0f, 8.0f};
        float32_t out[4] = {0};
        failures += check_f32_lane(
            "leaky_relu_f32", arm_nn_activation_f32(in32, out, 4, ARM_NN_FLT_ACT_LEAKY_RELU, alpha), out, expect);
    }

    {
        const float32_t expect[3] = {0.0f, 3.0f, 8.0f};
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane(
            "relu_f16", arm_nn_activation_f16(in16, out, 4, ARM_NN_FLT_ACT_RELU, (float16_t)0.0f), out, expect);
    }
    {
        const float32_t expect[3] = {0.0f, 3.0f, 6.0f};
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane(
            "relu6_f16", arm_nn_activation_f16(in16, out, 4, ARM_NN_FLT_ACT_RELU6, (float16_t)0.0f), out, expect);
    }
    {
        const float32_t expect[3] = {-1.0f, 3.0f, 8.0f};
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane("leaky_relu_f16",
                                   arm_nn_activation_f16(in16, out, 4, ARM_NN_FLT_ACT_LEAKY_RELU, (float16_t)alpha),
                                   out,
                                   expect);
    }

    {
        /* Tanh f16 (#407): the scalar LUT helper classifies NaN in the integer
         * domain, so it must propagate under this flag pair too. The finite
         * probes sit on exact grid points, where interpolation degenerates to a
         * table read, so the expectation is the exact LUT entry. */
        float32_t expect[3];
        _Float16 t2, t3;
        memcpy(&t2, &arm_nn_tanh_lut_f16[128], sizeof(t2)); /* tanh(2) sample */
        memcpy(&t3, &arm_nn_tanh_lut_f16[192], sizeof(t3)); /* tanh(3) sample */
        expect[0] = -(float32_t)t2;
        expect[1] = (float32_t)t3;
        expect[2] = 1.0f; /* 8 > 4 saturates exactly */
        float16_t out[4] = {(float16_t)0.0f};
        failures += check_f16_lane(
            "tanh_f16", arm_nn_activation_f16(in16, out, 4, ARM_NN_FLT_ACT_TANH, (float16_t)0.0f), out, expect);
    }

    if (failures == 0)
    {
        printf("activation -Ofast NaN probe: 7/7 scalar legs propagate\n");
    }
    return failures;
}
