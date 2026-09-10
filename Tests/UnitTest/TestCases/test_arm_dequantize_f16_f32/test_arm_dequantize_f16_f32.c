/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * arm_dequantize_f16_f32 (#411): bit-exact float16 -> float32 widening.
 *
 * Reference: for every non-NaN half the compiler's own (float32_t) cast, which is the NumPy
 * float32(float16(x)) semantics and is independent of the kernel (VCVT scalar on an FP16 core,
 * libgcc's software conversion elsewhere). For a NaN the contract is the bit-expansion
 * sign | 0x7F800000 | mantissa << 13, i.e. sign, quiet bit and payload preserved. Every comparison
 * is bit-for-bit, the output sits between guard words, and the exhaustive case covers all 65536
 * input patterns in blocks that are not multiples of the vector width.
 */

#include <arm_nnfunctions.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#define DQ_GUARD 8
#define DQ_MAX 1000
#define DQ_GUARD_BITS 0xA5A5A5A5u

static float16_t dq_in[DQ_MAX];
static float32_t dq_out[DQ_MAX + 2 * DQ_GUARD];

static uint32_t dq_bits32(float32_t x)
{
    uint32_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}
static float16_t dq_half(uint16_t bits)
{
    volatile uint16_t staged = bits;
    const uint16_t b = staged;
    float16_t h;
    memcpy(&h, &b, sizeof(h));
    return h;
}
static uint16_t dq_bits16(float16_t h)
{
    uint16_t b;
    memcpy(&b, &h, sizeof(b));
    return b;
}

static uint32_t dq_expected(uint16_t h)
{
    if ((h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0u)
    {
        return ((uint32_t)(h & 0x8000u) << 16) | 0x7F800000u | ((uint32_t)(h & 0x03FFu) << 13);
    }
    /* The cast is the reference; staged through volatile so it is evaluated at run time, by the
     * toolchain's conversion, not folded from the constant. */
    volatile float16_t v = dq_half(h);
    const float32_t f = (float32_t)v;
    return dq_bits32(f);
}

/* Runs one block of n halves already in dq_in and checks every output word and the guards. */
static void dq_run(int32_t n)
{
    for (int32_t i = 0; i < DQ_MAX + 2 * DQ_GUARD; i++)
    {
        memcpy(&dq_out[i], &(uint32_t){DQ_GUARD_BITS}, sizeof(uint32_t));
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_f32(dq_in, dq_out + DQ_GUARD, n));
    for (int32_t i = 0; i < DQ_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_HEX32(DQ_GUARD_BITS, dq_bits32(dq_out[i]));
        TEST_ASSERT_EQUAL_HEX32(DQ_GUARD_BITS, dq_bits32(dq_out[DQ_GUARD + n + i]));
    }
    for (int32_t i = 0; i < n; i++)
    {
        TEST_ASSERT_EQUAL_HEX32(dq_expected(dq_bits16(dq_in[i])), dq_bits32(dq_out[DQ_GUARD + i]));
    }
}

static uint32_t dq_seed = 0xD0D0u;
static uint16_t dq_rand16(void)
{
    dq_seed = dq_seed * 1664525u + 1013904223u;
    return (uint16_t)(dq_seed >> 16);
}

/* Block sizes 0, 1, 15, 16, 17, 1000 (and the other non-multiples of 8) over random halves. */
void dequantize_f16_f32_block_sizes(void)
{
    static const int32_t sizes[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 23, 31, 33, 1000};
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++)
    {
        for (int32_t i = 0; i < sizes[k]; i++)
        {
            dq_in[i] = dq_half(dq_rand16());
        }
        dq_run(sizes[k]);
    }
}

/* Every one of the 65536 half patterns, in blocks of 999 (not a multiple of 8). */
void dequantize_f16_f32_exhaustive(void)
{
    int32_t n = 0;
    for (uint32_t bits = 0; bits < 0x10000u; bits++)
    {
        dq_in[n++] = dq_half((uint16_t)bits);
        if (n == 999 || bits == 0xFFFFu)
        {
            dq_run(n);
            n = 0;
        }
    }
}

/* NaN lanes mixed with ordinary lanes in every position of an 8-lane block: the MVE lane repair must
 * touch only the NaN lanes. */
void dequantize_f16_f32_nan_lanes_mixed(void)
{
    static const uint16_t nans[] = {0x7E00u, 0xFE00u, 0x7C01u, 0xFDFFu, 0x7FFFu, 0x7E55u, 0xFC2Au};
    for (int32_t pos = 0; pos < 8; pos++)
    {
        for (int32_t i = 0; i < 64; i++)
        {
            dq_in[i] = dq_half((i % 8 == pos) ? nans[(size_t)(i / 8) % 7] : dq_rand16());
            /* keep the non-NaN lanes non-NaN */
            if (i % 8 != pos && (dq_bits16(dq_in[i]) & 0x7C00u) == 0x7C00u)
            {
                dq_in[i] = dq_half((uint16_t)(dq_bits16(dq_in[i]) & 0x7BFFu));
            }
        }
        dq_run(64);
        dq_run(61);
    }
}

/* All 1023 positive and negative subnormals plus +/-0 and the smallest normals. */
void dequantize_f16_f32_subnormals_and_zeros(void)
{
    int32_t n = 0;
    for (uint32_t m = 0; m <= 0x3FFu; m++)
    {
        dq_in[n++] = dq_half((uint16_t)m);
        dq_in[n++] = dq_half((uint16_t)(0x8000u | m));
        if (n >= 998)
        {
            dq_run(n);
            n = 0;
        }
    }
    dq_in[n++] = dq_half(0x0400u);
    dq_in[n++] = dq_half(0x8400u);
    dq_run(n);
}

/* Negative block size and NULL pointers with work to do: ARG_ERROR, output untouched. */
void dequantize_f16_f32_arg_error(void)
{
    for (int32_t i = 0; i < DQ_MAX + 2 * DQ_GUARD; i++)
    {
        memcpy(&dq_out[i], &(uint32_t){DQ_GUARD_BITS}, sizeof(uint32_t));
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_f16_f32(dq_in, dq_out + DQ_GUARD, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_f16_f32(NULL, dq_out + DQ_GUARD, 8));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_f16_f32(dq_in, NULL, 8));
    for (int32_t i = 0; i < DQ_MAX + 2 * DQ_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_HEX32(DQ_GUARD_BITS, dq_bits32(dq_out[i]));
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_f32(NULL, NULL, 0));
}
