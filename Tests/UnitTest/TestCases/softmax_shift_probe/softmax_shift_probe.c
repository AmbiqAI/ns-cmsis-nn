/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/* Built with shift-base checks (see CMakeLists.txt): the s8, s8_s16 and u8 softmax requantization must not
 * left-shift a negative or overflowing signed value (#704), and must keep the outputs pinned below. Long rows of
 * equal values must round 1/n correctly (#705, #710). */

static uint32_t lcg = 9u;
static int32_t rnd(int32_t lo, int32_t hi)
{
    lcg = lcg * 1664525u + 1013904223u;
    return lo + (int32_t)((lcg >> 8) % (uint32_t)(hi - lo + 1));
}

static uint32_t fnv(uint32_t h, uint32_t v) { return (h ^ v) * 16777619u; }

int main(void)
{
    static int8_t in[3 * 200];
    static uint8_t in_u8[3 * 200];
    static int8_t out[3 * 200];
    static int16_t out16[3 * 200];
    static uint8_t out_u8[3 * 200];
    const int32_t cols[] = {5, 12, 99, 200};
    const int32_t ranges[][2] = {{-128, 127}, {120, 127}, {-128, -120}};
    uint32_t h = 2166136261u;
    int failures = 0;

    for (size_t c = 0; c < sizeof(cols) / sizeof(cols[0]); c++)
    {
        for (size_t r = 0; r < sizeof(ranges) / sizeof(ranges[0]); r++)
        {
            const int32_t n = 3 * cols[c];
            for (int32_t i = 0; i < n; i++)
            {
                in[i] = (int8_t)rnd(ranges[r][0], ranges[r][1]);
                in_u8[i] = (uint8_t)(in[i] + 128);
            }
            arm_softmax_s8(in, 3, cols[c], 1077952576, 23, -248, out);
            arm_softmax_s8_s16(in, 3, cols[c], 1077952576, 23, -248, out16);
            arm_softmax_u8(in_u8, 3, cols[c], 1077952576, 23, -248, out_u8);
            for (int32_t i = 0; i < n; i++)
            {
                h = fnv(h, (uint8_t)out[i]);
                h = fnv(h, (uint16_t)out16[i]);
                h = fnv(h, out_u8[i]);
            }
        }
    }

    /* A positive diff_min leaves every element out, so the row sum is 0 */
    arm_softmax_s8(in, 1, 12, 1077952576, 23, 1, out);
    arm_softmax_s8_s16(in, 1, 12, 1077952576, 23, 1, out16);
    arm_softmax_u8(in_u8, 1, 12, 1077952576, 23, 1, out_u8);
    for (int32_t i = 0; i < 12; i++)
    {
        if (out[i] != INT8_MIN || out16[i] != INT16_MIN || out_u8[i] != 0)
        {
            printf("empty row: output %ld is %d, %d, %u\n", (long)i, out[i], out16[i], out_u8[i]);
            failures++;
        }
    }

    /* Rows of n equal values: from 256 the final divide reaches 2^31, from 512 the quotient rounds to 0, and from
       4096 the row sum passes int32_t */
    static int8_t long_in[4096];
    static uint8_t long_in_u8[4096];
    static int8_t long_out[4096];
    static int16_t long_out16[4096];
    static uint8_t long_out_u8[4096];
    const int32_t sizes[] = {256, 601, 4096};
    const int8_t want[] = {-127, -128, -128};
    const int16_t want16[] = {-32512, -32659, -32752};
    const uint8_t want_u8[] = {1, 0, 0};
    memset(long_in, 5, sizeof(long_in));
    memset(long_in_u8, 133, sizeof(long_in_u8));
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++)
    {
        arm_softmax_s8(long_in, 1, sizes[k], 1077952576, 23, -248, long_out);
        arm_softmax_s8_s16(long_in, 1, sizes[k], 1077952576, 23, -248, long_out16);
        arm_softmax_u8(long_in_u8, 1, sizes[k], 1077952576, 23, -248, long_out_u8);
        for (int32_t i = 0; i < sizes[k]; i++)
        {
            if (long_out[i] != want[k] || long_out16[i] != want16[k] || long_out_u8[i] != want_u8[k])
            {
                printf("row of %ld: output %ld is %d, %d, %u\n",
                       (long)sizes[k],
                       (long)i,
                       long_out[i],
                       long_out16[i],
                       long_out_u8[i]);
                failures++;
                break;
            }
        }
    }

    /* The helpers themselves, at negative and saturating inputs: a power-of-two multiply against a 64-bit
       saturating reference, and the exponential across its input range. */
    const int32_t vals[] = {INT32_MIN, -1073741825, -65536, -1, 0, 1, 65535, 1073741824, INT32_MAX};
    for (int32_t exp = 1; exp <= 30; exp++)
    {
        for (size_t v = 0; v < sizeof(vals) / sizeof(vals[0]); v++)
        {
            int64_t ref = (int64_t)vals[v] * ((int64_t)1 << exp);
            ref = ref > INT32_MAX ? INT32_MAX : (ref < INT32_MIN ? INT32_MIN : ref);
            const int32_t got = arm_nn_mult_by_power_of_two(vals[v], exp);
            if ((int64_t)got != ref)
            {
                printf("mult_by_power_of_two(%ld, %ld) = %ld, expected %lld\n",
                       (long)vals[v],
                       (long)exp,
                       (long)got,
                       (long long)ref);
                failures++;
            }
        }
    }
    for (int64_t v = 0; v >= INT32_MIN; v -= 4194301)
    {
        h = fnv(h, (uint32_t)arm_nn_exp_on_negative_values((int32_t)v));
    }

    const uint32_t expected = 0x03492351u;
    if (h != expected)
    {
        printf("softmax_shift_probe: output hash %08lx, expected %08lx\n", (unsigned long)h, (unsigned long)expected);
        failures++;
    }
    printf("softmax_shift_probe: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
