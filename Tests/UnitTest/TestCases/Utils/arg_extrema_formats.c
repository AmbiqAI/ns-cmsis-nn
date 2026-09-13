/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "arm_nnfunctions.h"
#include <stdint.h>
#include <string.h>

/* Compile with the real scalar half-format option; no synthetic format macros. */
#if defined(__ARM_FP16_FORMAT_ALTERNATIVE)
static const float16_t encoded[] = {65600.0f, 65504.0f};
#else
static const float16_t encoded[] = {65504.0f, 1.0f};
#endif

/* Exact finite decoding in units of 2^-24; independent of the kernel's keys. */
static int64_t value(uint16_t bits)
{
    uint32_t exponent = (bits >> 10) & 31;
    uint32_t fraction = bits & 1023;
#if !defined(__ARM_FP16_FORMAT_ALTERNATIVE)
    if (exponent == 31)
        return (bits & 0x8000) ? -INT64_MAX : INT64_MAX;
#endif
    int64_t magnitude = exponent ? (int64_t)(1024 + fraction) << (exponent - 1) : fraction;
    return (bits & 0x8000) ? -magnitude : magnitude;
}

static int nan_bits(uint16_t bits)
{
#if defined(__ARM_FP16_FORMAT_ALTERNATIVE)
    (void)bits;
    return 0;
#else
    return (bits & 0x7fff) > 0x7c00;
#endif
}

int test_formats(void)
{
    cmsis_nn_dims dims = {1, 1, 1, 2};
    int32_t output;
    uint16_t encoding[2];
    memcpy(encoding, encoded, sizeof(encoding));
#if defined(__ARM_FP16_FORMAT_ALTERNATIVE)
    if (encoding[0] != 0x7c01 || encoding[1] != 0x7bff)
        return 1;
#else
    if (encoding[0] != 0x7bff || encoding[1] != 0x3c00)
        return 2;
#endif
    if (arm_argmin_f16(encoded, &dims, 3, &output) != ARM_CMSIS_NN_SUCCESS || output != 1)
        return 3;
    if (arm_argmax_f16(encoded, &dims, 3, &output) != ARM_CMSIS_NN_SUCCESS || output != 0)
        return 4;

    const uint16_t anchors[] = {0, 0x8000, 0x7bff, 0x7c00, 0x7c01, 0x7fff, 0xfc00, 0xffff};
    for (uint32_t bits = 0; bits < 65536; ++bits)
        for (unsigned a = 0; a < sizeof(anchors) / sizeof(anchors[0]); ++a)
        {
            const uint16_t words[2] = {(uint16_t)bits, anchors[a]};
            float16_t input[2];
            memcpy(input, words, sizeof(input));
            int32_t minimum = nan_bits(words[0]) ? 0 : nan_bits(words[1]) ? 1 : value(words[1]) < value(words[0]);
            int32_t maximum = nan_bits(words[0]) ? 0 : nan_bits(words[1]) ? 1 : value(words[1]) > value(words[0]);
            if (arm_argmin_f16(input, &dims, 3, &output) != ARM_CMSIS_NN_SUCCESS || output != minimum)
                return 5;
            if (arm_argmax_f16(input, &dims, 3, &output) != ARM_CMSIS_NN_SUCCESS || output != maximum)
                return 6;
        }
#if ARM_NN_ENABLE_F32
    /* Alternative half must not disable IEEE single-precision NaN selection. */
    const uint32_t words32[] = {0x3f800000, 0x7f800001};
    float32_t input32[2];
    memcpy(input32, words32, sizeof(input32));
    if (arm_argmin_f32(input32, &dims, 3, &output) != ARM_CMSIS_NN_SUCCESS || output != 1)
        return 7;
    if (arm_argmax_f32(input32, &dims, 3, &output) != ARM_CMSIS_NN_SUCCESS || output != 1)
        return 8;
#endif
    return 0;
}
