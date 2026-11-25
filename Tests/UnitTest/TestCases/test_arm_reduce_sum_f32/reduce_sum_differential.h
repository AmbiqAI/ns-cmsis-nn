/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The differential cases: arm_reduce_sum_f32 against the body it had (reduce_sum_previous.h), on
 * the same input, each writing into its own guarded output. A case differs when the statuses
 * differ, when any byte of the two outputs differs (guards included), when a guard of the
 * entry's output moved, or when the input changed. Every buffer is prepared and compared over
 * what the case can touch, the larger of the input's element count and the output shape's
 * (the suffix branch writes one sum per row of the input whatever the output shape says), and
 * RD_GUARD words either side.
 *
 * The entry refuses only missing pointers: a shape whose element count passes INT32_MAX is not
 * refused but leaves the spatial branch for the suffix or generic one, which would read it, so
 * no such shape can run here. The spatial branch's own decision is held apart from any shape by
 * rd_fits, below. */

#include "reduce_sum_previous.h"
#include <stdio.h>
#include <string.h>
#include <unity.h>

#define RD_CAP 2304
#define RD_GUARD 16
#define RD_BYTE 0xa5
#define RD_REPORTED 8

static uint32_t rd_bits[RD_CAP];
static float32_t rd_input[RD_CAP + 2 * RD_GUARD];
static float32_t rd_new[RD_CAP + 2 * RD_GUARD];
static float32_t rd_old[RD_CAP + 2 * RD_GUARD];
static unsigned long rd_cases;
static unsigned long rd_differences;
static char rd_message[192];

/* The output count the shape and the mask give where every extent is non-negative, the input's
 * count otherwise bounded by the caller; an extent below zero contributes no element. */
static long rd_count(const int32_t d[4])
{
    long long count = 1;
    for (int k = 0; k < 4; ++k)
        count *= d[k] > 0 ? d[k] : 0;
    return (long)count;
}

static void rd_case(const char *pattern,
                    const int32_t d[4],
                    const int32_t axes[4],
                    const int32_t out[4],
                    int pass_input,
                    int pass_input_dims,
                    int pass_axes,
                    int pass_output,
                    int pass_output_dims)
{
    const cmsis_nn_dims input_dims = {d[0], d[1], d[2], d[3]};
    const cmsis_nn_dims axis_dims = {axes[0], axes[1], axes[2], axes[3]};
    const cmsis_nn_dims output_dims = {out[0], out[1], out[2], out[3]};
    const size_t count = (size_t)rd_count(d);
    const size_t written = (size_t)rd_count(out);
    const size_t in_words = count + 2 * RD_GUARD;
    const size_t out_words = (written > count ? written : count) + 2 * RD_GUARD;
    memset(rd_input, RD_BYTE, in_words * sizeof(float32_t));
    memcpy(rd_input + RD_GUARD, rd_bits, count * sizeof(uint32_t));
    memset(rd_new, RD_BYTE, out_words * sizeof(float32_t));
    memset(rd_old, RD_BYTE, out_words * sizeof(float32_t));
    const float32_t *input = pass_input ? rd_input + RD_GUARD : NULL;
    const arm_cmsis_nn_status got = arm_reduce_sum_f32(input,
                                                       pass_input_dims ? &input_dims : NULL,
                                                       pass_axes ? &axis_dims : NULL,
                                                       pass_output ? rd_new + RD_GUARD : NULL,
                                                       pass_output_dims ? &output_dims : NULL);
    const arm_cmsis_nn_status want = rp_reduce_sum_f32(input,
                                                       pass_input_dims ? &input_dims : NULL,
                                                       pass_axes ? &axis_dims : NULL,
                                                       pass_output ? rd_old + RD_GUARD : NULL,
                                                       pass_output_dims ? &output_dims : NULL);
    int differs = got != want;
    differs |= memcmp(rd_new, rd_old, out_words * sizeof(float32_t)) != 0;
    /* The guards of the entry's output, and the input, on their own. */
    const unsigned char *bytes = (const unsigned char *)rd_new;
    for (size_t i = 0; i < out_words * sizeof(float32_t); ++i)
    {
        const int inside = got == ARM_CMSIS_NN_SUCCESS && i >= RD_GUARD * sizeof(float32_t) &&
            i < (RD_GUARD + (written > count ? written : count)) * sizeof(float32_t);
        if (!inside && bytes[i] != RD_BYTE)
            differs = 1;
    }
    const unsigned char *in_bytes = (const unsigned char *)rd_input;
    for (size_t i = 0; i < in_words * sizeof(float32_t); ++i)
    {
        const int inside = i >= RD_GUARD * sizeof(float32_t) && i < (RD_GUARD + count) * sizeof(float32_t);
        if (!inside && in_bytes[i] != RD_BYTE)
            differs = 1;
    }
    differs |= memcmp(rd_input + RD_GUARD, rd_bits, count * sizeof(uint32_t)) != 0;
    ++rd_cases;
    if (differs)
    {
        if (rd_differences < RD_REPORTED)
        {
            snprintf(rd_message,
                     sizeof(rd_message),
                     "difference: %s dims {%ld, %ld, %ld, %ld} axes {%ld, %ld, %ld, %ld} status %d against %d",
                     pattern,
                     (long)d[0],
                     (long)d[1],
                     (long)d[2],
                     (long)d[3],
                     (long)axes[0],
                     (long)axes[1],
                     (long)axes[2],
                     (long)axes[3],
                     (int)got,
                     (int)want);
            TEST_MESSAGE(rd_message);
        }
        ++rd_differences;
    }
}

static uint32_t rd_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static uint32_t rd_bits_of(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* Patterns: approximately normal values; random encodings (every class: NaN payloads of both
 * signs, infinities, subnormals); specials only (both zeros, both infinities, quiet and
 * signalling NaNs, the smallest subnormals, one); and large values of both signs among small
 * ones, whose sums change with the order of the additions. */
#define RD_PATTERNS 4
static const char *const rd_pattern_names[RD_PATTERNS] = {"normals", "random bits", "specials", "cancellation"};
static const uint32_t rd_specials[] = {0,
                                       UINT32_C(0x80000000),
                                       UINT32_C(0x7f800000),
                                       UINT32_C(0xff800000),
                                       UINT32_C(0x7fc00000),
                                       UINT32_C(0xffc00001),
                                       UINT32_C(0x7f800001),
                                       UINT32_C(0xffbfffff),
                                       UINT32_C(0x00000001),
                                       UINT32_C(0x80000001),
                                       UINT32_C(0x00800000),
                                       UINT32_C(0x3f800000)};
#define RD_SPECIALS ((int)(sizeof(rd_specials) / sizeof(rd_specials[0])))

static void rd_fill(int pattern, size_t count, uint32_t *state)
{
    for (size_t i = 0; i < count; ++i)
    {
        const uint32_t r = rd_next(state);
        switch (pattern)
        {
        case 0: {
            float32_t sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += (float32_t)(rd_next(state) >> 8) * (1.0f / 16777216.0f);
            rd_bits[i] = rd_bits_of(sum - 2.0f);
            break;
        }
        case 1:
            rd_bits[i] = r;
            break;
        case 2:
            rd_bits[i] = rd_specials[r % RD_SPECIALS];
            break;
        default: {
            const float32_t big = (r & 1) ? 1.0e30f : -1.0e30f;
            rd_bits[i] = (r % 5) == 0 ? rd_bits_of(big) : rd_bits_of((float32_t)((r >> 8) & 255) * 0.25f);
            break;
        }
        }
    }
}

static void rd_report(const char *what, unsigned long before_cases, unsigned long before_differences)
{
    snprintf(rd_message,
             sizeof(rd_message),
             "differential %s: %lu cases, %lu differences",
             what,
             rd_cases - before_cases,
             rd_differences - before_differences);
    TEST_MESSAGE(rd_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(rd_differences - before_differences));
}

/* The extents of the shape sweep: one, small, a vector's width on each side, and 127, 128 and
 * 129 on each side of the small-extent gate. */
static const int32_t rd_extents[] = {1, 2, 3, 4, 5, 7, 8, 127, 128, 129};
#define RD_EXTENTS ((int)(sizeof(rd_extents) / sizeof(rd_extents[0])))

/* The output shape of a mask: a reduced extent becomes 1. */
static void rd_output(const int32_t d[4], const int32_t axes[4], int32_t out[4])
{
    for (int k = 0; k < 4; ++k)
        out[k] = axes[k] ? 1 : d[k];
}

/* Every shape over rd_extents whose element count is at most RD_CAP, under every one of the 16
 * masks (set fields written as 1, 2 or -1 in turn), each with one pattern in turn; and, on an
 * Arm target with an FPU, the same sweep again with flush-to-zero set and round-to-nearest, the
 * control the Helium spatial body is chosen under. */
static void rd_sweep(const char *what, uint32_t seed)
{
    const unsigned long before_cases = rd_cases;
    const unsigned long before_differences = rd_differences;
    uint32_t state = seed;
    int turn = 0;
    for (int a = 0; a < RD_EXTENTS; ++a)
        for (int b = 0; b < RD_EXTENTS; ++b)
            for (int c = 0; c < RD_EXTENTS; ++c)
                for (int e = 0; e < RD_EXTENTS; ++e)
                {
                    const int32_t d[4] = {rd_extents[a], rd_extents[b], rd_extents[c], rd_extents[e]};
                    if (rd_count(d) > RD_CAP)
                        continue;
                    for (int mask = 0; mask < 16; ++mask)
                    {
                        const int32_t set = turn % 3 == 0 ? 1 : turn % 3 == 1 ? 2 : -1;
                        const int32_t axes[4] = {
                            (mask & 8) ? set : 0, (mask & 4) ? set : 0, (mask & 2) ? set : 0, (mask & 1) ? set : 0};
                        int32_t out[4];
                        rd_output(d, axes, out);
                        const int pattern = turn++ % RD_PATTERNS;
                        rd_fill(pattern, (size_t)rd_count(d), &state);
                        rd_case(rd_pattern_names[pattern], d, axes, out, 1, 1, 1, 1, 1);
                    }
                }
    rd_report(what, before_cases, before_differences);
}

static void rd_shapes(void) { rd_sweep("shapes", UINT32_C(0x2545f491)); }

static void rd_shapes_flush(void)
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
    uint32_t saved, restored;
    __asm volatile("vmrs %0, fpscr" : "=r"(saved) : : "memory");
    const uint32_t controlled = (saved & ~((3u << 22) | (1u << 26))) | (1u << 24);
    __asm volatile("vmsr fpscr, %0" : : "r"(controlled) : "memory");
    rd_sweep("shapes under flush-to-zero", UINT32_C(0x51f15e1d));
    __asm volatile("vmsr fpscr, %0" : : "r"(saved) : "memory");
    __asm volatile("vmrs %0, fpscr" : "=r"(restored) : : "memory");
    TEST_ASSERT_EQUAL_HEX32(saved, restored);
#else
    TEST_IGNORE_MESSAGE("Requires an Arm FPU's FPSCR");
#endif
}

/* The entry's one refusal and the shapes outside the spatial branch's domain: each pointer
 * missing, on a spatial and a non-spatial mask; zero and negative extents at each position
 * under every mask; and output shapes that do not match the mask (each kept extent one
 * smaller), which leave the spatial branch for the generic one. */
static void rd_refusals(void)
{
    const unsigned long before_cases = rd_cases;
    const unsigned long before_differences = rd_differences;
    uint32_t state = UINT32_C(0x9e3779b9);
    rd_fill(1, RD_CAP, &state);
    const int32_t d[4] = {2, 3, 4, 5};
    const int32_t masks[3][4] = {{0, 1, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};
    for (int m = 0; m < 3; ++m)
    {
        int32_t out[4];
        rd_output(d, masks[m], out);
        for (int missing = 0; missing < 5; ++missing)
            rd_case("missing pointer",
                    d,
                    masks[m],
                    out,
                    missing != 0,
                    missing != 1,
                    missing != 2,
                    missing != 3,
                    missing != 4);
    }
    const int32_t odd[] = {0, -1, -2, -128};
    for (int position = 0; position < 4; ++position)
        for (unsigned i = 0; i < sizeof(odd) / sizeof(odd[0]); ++i)
            for (int mask = 0; mask < 16; ++mask)
            {
                int32_t shape[4] = {2, 3, 1, 2};
                shape[position] = odd[i];
                const int32_t axes[4] = {(mask >> 3) & 1, (mask >> 2) & 1, (mask >> 1) & 1, mask & 1};
                int32_t out[4];
                rd_output(shape, axes, out);
                rd_fill(0, (size_t)rd_count(shape), &state);
                rd_case("zero or negative extent", shape, axes, out, 1, 1, 1, 1, 1);
            }
    for (int mask = 0; mask < 16; ++mask)
    {
        const int32_t axes[4] = {(mask >> 3) & 1, (mask >> 2) & 1, (mask >> 1) & 1, mask & 1};
        for (int position = 0; position < 4; ++position)
        {
            if (axes[position])
                continue;
            int32_t out[4];
            rd_output(d, axes, out);
            out[position] -= 1;
            rd_fill(0, (size_t)rd_count(d), &state);
            rd_case("output shape", d, axes, out, 1, 1, 1, 1, 1);
        }
    }
    rd_report("refusals", before_cases, before_differences);
}

/* The small-extent gate's premise, on the previous bound alone (rp_reduce_sum_spatial_fits): it
 * accepts every shape of four extents in [1, 127], the domain where arm_reduce_sum_f32 now skips
 * it, here every 4-tuple over 1 to 5, 63 to 65 and 126 to 127 (127^4 is the largest count of the
 * domain and the bound is monotone in each extent); and just outside the gate, where the bound
 * still runs, it accepts and refuses as the count says (128^4 = 2^28 accepted; 128^3 x 1024 =
 * 2^31 refused; 46341^2 refused, 46340^2 accepted). The entry's other side, the bound itself, is
 * the previous loop unchanged. */
static void rd_fits(void)
{
    const int32_t inside[] = {1, 2, 3, 4, 5, 63, 64, 65, 126, 127};
    const int n = (int)(sizeof(inside) / sizeof(inside[0]));
    unsigned long cases = 0, refused = 0;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c)
                for (int e = 0; e < n; ++e)
                {
                    const cmsis_nn_dims dims = {inside[a], inside[b], inside[c], inside[e]};
                    ++cases;
                    refused += !rp_reduce_sum_spatial_fits(&dims);
                }
    const struct
    {
        cmsis_nn_dims dims;
        bool fits;
    } outside[] = {{{128, 128, 128, 128}, true},
                   {{128, 128, 128, 1024}, false},
                   {{128, 128, 128, 1023}, true},
                   {{46341, 46341, 1, 1}, false},
                   {{46340, 46340, 1, 1}, true},
                   {{1, 1, 1, INT32_MAX}, true},
                   {{2, 1, 1, INT32_C(1073741824)}, false},
                   {{128, 1, 1, 1}, true}};
    unsigned long wrong = 0;
    for (unsigned i = 0; i < sizeof(outside) / sizeof(outside[0]); ++i)
        wrong += rp_reduce_sum_spatial_fits(&outside[i].dims) != outside[i].fits;
    snprintf(rd_message,
             sizeof(rd_message),
             "differential fits: %lu shapes inside the gate, %lu refused; %u outside, %lu decided otherwise",
             cases,
             refused,
             (unsigned)(sizeof(outside) / sizeof(outside[0])),
             wrong);
    TEST_MESSAGE(rd_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)refused);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)wrong);
}
