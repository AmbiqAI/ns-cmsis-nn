/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The differential cases: arm_argmax_f32 and arm_argmin_f32, and where binary16 is built
 * arm_argmax_f16 and arm_argmin_f16 (all four share Include/Internal/arm_nn_arg_extrema_flt.h),
 * against the body they share as it was (arg_extrema_previous.h), on the same input, each
 * writing into its own guarded output. A case differs when the statuses differ, when any byte
 * of the two outputs differs (guards included), when a guard of the entry's output moved, or
 * when the input changed. Every buffer is checked over what the case can touch and AD_GUARD
 * words either side. */

#include "../../../../Include/Internal/arm_nn_arg_extrema_flt.h"
#include "arg_extrema_previous.h"
#include <stdio.h>
#include <string.h>
#include <unity.h>

#define AD_CAP 2304
#define AD_GUARD 16
#define AD_BYTE 0xa5
#define AD_SENTINEL INT32_C(0x5a5a5a5a)
#define AD_REPORTED 8

#define AD_F32_INF UINT32_C(0x7f800000)
#define AD_F16_INF UINT32_C(0x7c00)

static float32_t ad_input32[AD_CAP + 2 * AD_GUARD];
#if ARM_NN_ENABLE_F16
static float16_t ad_input16[AD_CAP + 2 * AD_GUARD];
#endif
static uint32_t ad_bits[AD_CAP];
static int32_t ad_new[AD_CAP + 2 * AD_GUARD];
static int32_t ad_old[AD_CAP + 2 * AD_GUARD];
static unsigned long ad_cases;
static unsigned long ad_differences;
static unsigned long ad_decided_only;
static char ad_message[192];

/* The entries: binary32 maximum and minimum, then binary16. */
#if ARM_NN_ENABLE_F16
    #define AD_ENTRIES 4
#else
    #define AD_ENTRIES 2
#endif
static const char *const ad_names[4] = {"argmax_f32", "argmin_f32", "argmax_f16", "argmin_f16"};

static size_t ad_width(int entry) { return entry < 2 ? sizeof(float32_t) : sizeof(uint16_t); }

static bool ad_maximum(int entry) { return (entry & 1) == 0; }

static arm_cmsis_nn_status ad_call(int entry, int pass_input, const cmsis_nn_dims *dims, int32_t axis, int32_t *out)
{
    switch (entry)
    {
    case 0:
        return arm_argmax_f32(pass_input ? ad_input32 + AD_GUARD : NULL, dims, axis, out);
    case 1:
        return arm_argmin_f32(pass_input ? ad_input32 + AD_GUARD : NULL, dims, axis, out);
#if ARM_NN_ENABLE_F16
    case 2:
        return arm_argmax_f16(pass_input ? ad_input16 + AD_GUARD : NULL, dims, axis, out);
    case 3:
        return arm_argmin_f16(pass_input ? ad_input16 + AD_GUARD : NULL, dims, axis, out);
#endif
    default:
        return ARM_CMSIS_NN_ARG_ERROR;
    }
}

static const void *ad_input(int entry, int pass_input)
{
    if (!pass_input)
        return NULL;
#if ARM_NN_ENABLE_F16
    if (entry >= 2)
        return ad_input16 + AD_GUARD;
#endif
    (void)entry;
    return ad_input32 + AD_GUARD;
}

static unsigned char *ad_input_bytes(int entry)
{
#if ARM_NN_ENABLE_F16
    if (entry >= 2)
        return (unsigned char *)ad_input16;
#endif
    (void)entry;
    return (unsigned char *)ad_input32;
}

/* The elements a shape holds when every extent is in [0, AD_CAP] and their product is at most
 * AD_CAP; -1 otherwise (the case then prepares and compares the whole buffers). */
static long ad_small_count(const cmsis_nn_dims *dims)
{
    if (!dims || dims->n < 0 || dims->h < 0 || dims->w < 0 || dims->c < 0 || dims->n > AD_CAP || dims->h > AD_CAP ||
        dims->w > AD_CAP || dims->c > AD_CAP)
        return -1;
    const long long count = (long long)dims->n * dims->h * dims->w * dims->c;
    return count <= AD_CAP ? (long)count : -1;
}

/* outer * inner * reduction, or UINT64_MAX where it passes 64 bits. */
static uint64_t ad_product3(size_t outer, size_t inner, size_t reduction)
{
    const uint64_t lines = (uint64_t)outer * (uint64_t)inner;
    if (outer && lines / outer != inner)
        return UINT64_MAX;
    if (reduction && lines > UINT64_MAX / reduction)
        return UINT64_MAX;
    return lines * (uint64_t)reduction;
}

/* One case: entry `entry` and the previous body on ad_bits[0, count) (count elements of the
 * entry's width, the low half of each word for binary16). A shape whose elements exceed
 * AD_CAP is run only where the previous chain refuses it, so neither body reads past the
 * buffer; one it accepts is left to the decision cases (ad_decided_only counts them). Before
 * a run, the library's own decision (arm_nn_arg_extrema_lines) is held to the shape: a body
 * that would accept a shape the previous chain refuses, or plan more elements than the shape
 * holds, differs without being run, so a wrong body fails the case instead of reading past
 * the buffers. */
static void
ad_case(const char *pattern, int entry, const cmsis_nn_dims *dims, int32_t axis, int pass_input, int pass_output)
{
    const size_t width = ad_width(entry);
    const long small = ad_small_count(dims);
    size_t plan_outer = 0, plan_inner = 0, plan_reduction = 0;
    const int plans_lines = arm_nn_arg_extrema_lines(dims, axis, width, &plan_outer, &plan_inner, &plan_reduction) ==
            ARM_CMSIS_NN_SUCCESS &&
        (uint64_t)plan_outer * (uint64_t)plan_inner != 0;
    int unsafe = 0;
    if (small < 0)
    {
        size_t outer, inner, reduction;
        uint64_t lines;
        if (ap_arg_lines(dims, axis, width, &outer, &inner, &reduction, &lines) == ARM_CMSIS_NN_SUCCESS && lines != 0)
        {
            if (plans_lines)
            {
                ++ad_decided_only;
                return;
            }
            unsafe = 1;
        }
        else
            unsafe = plans_lines;
    }
    else
        unsafe = plans_lines && ad_product3(plan_outer, plan_inner, plan_reduction) != (uint64_t)small;
    if (unsafe)
    {
        ++ad_cases;
        if (ad_differences < AD_REPORTED)
        {
            snprintf(ad_message,
                     sizeof(ad_message),
                     "difference: %s %s dims {%ld, %ld, %ld, %ld} axis %ld planned against the previous chain, not run",
                     ad_names[entry],
                     pattern,
                     dims ? (long)dims->n : -1L,
                     dims ? (long)dims->h : -1L,
                     dims ? (long)dims->w : -1L,
                     dims ? (long)dims->c : -1L,
                     (long)axis);
            TEST_MESSAGE(ad_message);
        }
        ++ad_differences;
        return;
    }
    const size_t count = small < 0 ? AD_CAP : (size_t)small;
    const size_t words = count + 2 * AD_GUARD;
    unsigned char *bytes = ad_input_bytes(entry);
    memset(bytes, AD_BYTE, words * width);
    for (size_t i = 0; i < count; ++i)
    {
        if (width == sizeof(float32_t))
            memcpy(bytes + (AD_GUARD + i) * width, &ad_bits[i], sizeof(uint32_t));
        else
        {
            const uint16_t half = (uint16_t)ad_bits[i];
            memcpy(bytes + (AD_GUARD + i) * width, &half, sizeof(half));
        }
    }
    for (size_t i = 0; i < words; ++i)
    {
        ad_new[i] = AD_SENTINEL;
        ad_old[i] = AD_SENTINEL;
    }
    const arm_cmsis_nn_status got = ad_call(entry, pass_input, dims, axis, pass_output ? ad_new + AD_GUARD : NULL);
    const arm_cmsis_nn_status want = ap_arg_extrema(
        ad_input(entry, pass_input), dims, axis, pass_output ? ad_old + AD_GUARD : NULL, width, ad_maximum(entry));
    int differs = got != want;
    differs |= memcmp(ad_new, ad_old, words * sizeof(int32_t)) != 0;
    /* The entry's guards and the input, on their own, so a fault both bodies share is caught too. */
    size_t written = 0;
    if (got == ARM_CMSIS_NN_SUCCESS && small > 0 && axis >= 0 && axis < 4)
    {
        const int32_t d[4] = {dims->n, dims->h, dims->w, dims->c};
        written = (size_t)small / (size_t)d[axis];
    }
    for (size_t i = 0; i < words; ++i)
    {
        const int inside = i >= AD_GUARD && i < AD_GUARD + written;
        if (!inside && ad_new[i] != AD_SENTINEL)
            differs = 1;
        if (inside &&
            (ad_new[i] < 0 ||
             (axis >= 0 && axis < 4 && dims &&
              ad_new[i] >= (axis == 0       ? dims->n
                                : axis == 1 ? dims->h
                                : axis == 2 ? dims->w
                                            : dims->c))))
            differs = 1;
    }
    for (size_t i = 0; i < words * width; ++i)
    {
        const int inside = i >= AD_GUARD * width && i < (AD_GUARD + count) * width;
        if (!inside && bytes[i] != AD_BYTE)
            differs = 1;
    }
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t now = 0;
        memcpy(&now, bytes + (AD_GUARD + i) * width, width);
        if (width == sizeof(uint16_t))
            now &= UINT32_C(0xffff);
        if (now != (width == sizeof(uint16_t) ? (ad_bits[i] & UINT32_C(0xffff)) : ad_bits[i]))
            differs = 1;
    }

    ++ad_cases;
    if (differs)
    {
        if (ad_differences < AD_REPORTED)
        {
            snprintf(ad_message,
                     sizeof(ad_message),
                     "difference: %s %s dims {%ld, %ld, %ld, %ld} axis %ld status %d against %d",
                     ad_names[entry],
                     pattern,
                     dims ? (long)dims->n : -1L,
                     dims ? (long)dims->h : -1L,
                     dims ? (long)dims->w : -1L,
                     dims ? (long)dims->c : -1L,
                     (long)axis,
                     (int)got,
                     (int)want);
            TEST_MESSAGE(ad_message);
        }
        ++ad_differences;
    }
}

static uint32_t ad_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

/* Special encodings of each width: both zeros, both infinities, NaNs of both signs (quiet and
 * signalling, smallest and largest payloads), the smallest subnormals, one and minus one. */
static const uint32_t ad_special32[] = {0,
                                        UINT32_C(0x80000000),
                                        AD_F32_INF,
                                        UINT32_C(0x80000000) | AD_F32_INF,
                                        AD_F32_INF | UINT32_C(0x00400000),
                                        UINT32_C(0x80000000) | AD_F32_INF | UINT32_C(0x00400000),
                                        AD_F32_INF | 1,
                                        UINT32_C(0x80000000) | AD_F32_INF | UINT32_C(0x007fffff),
                                        1,
                                        UINT32_C(0x80000001),
                                        UINT32_C(0x3f800000),
                                        UINT32_C(0xbf800000)};
static const uint32_t ad_special16[] = {0,
                                        UINT32_C(0x8000),
                                        AD_F16_INF,
                                        UINT32_C(0x8000) | AD_F16_INF,
                                        AD_F16_INF | UINT32_C(0x0200),
                                        UINT32_C(0x8000) | AD_F16_INF | UINT32_C(0x0200),
                                        AD_F16_INF | 1,
                                        UINT32_C(0x8000) | AD_F16_INF | UINT32_C(0x03ff),
                                        1,
                                        UINT32_C(0x8001),
                                        UINT32_C(0x3c00),
                                        UINT32_C(0xbc00)};
#define AD_SPECIALS ((int)(sizeof(ad_special32) / sizeof(ad_special32[0])))
#define AD_PATTERNS 6
static const char *const ad_pattern_names[AD_PATTERNS] =
    {"random bits", "special mix", "all NaN", "all equal", "zeros", "infinities and NaN"};

/* Fill ad_bits[0, count) with pattern `pattern` for an entry of `width` bytes. Random bits
 * reach every class of encoding; the special mix draws only the specials, so ties, NaNs and
 * zeros of both signs meet at every position; all NaN mixes payloads and signs. */
static void ad_fill(int pattern, size_t width, size_t count, uint32_t *state)
{
    const uint32_t *special = width == sizeof(float32_t) ? ad_special32 : ad_special16;
    const uint32_t mask = width == sizeof(float32_t) ? UINT32_MAX : UINT32_C(0xffff);
    for (size_t i = 0; i < count; ++i)
    {
        const uint32_t r = ad_next(state);
        switch (pattern)
        {
        case 0:
            ad_bits[i] = r & mask;
            break;
        case 1:
            ad_bits[i] = special[r % AD_SPECIALS];
            break;
        case 2:
            ad_bits[i] = special[4 + r % 4];
            break;
        case 3:
            ad_bits[i] = special[10];
            break;
        case 4:
            ad_bits[i] = special[r & 1];
            break;
        default:
            ad_bits[i] = special[2 + r % 6];
            break;
        }
    }
}

static void ad_report(const char *what, unsigned long before_cases, unsigned long before_differences)
{
    snprintf(ad_message,
             sizeof(ad_message),
             "differential %s: %lu cases, %lu differences",
             what,
             ad_cases - before_cases,
             ad_differences - before_differences);
    TEST_MESSAGE(ad_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(ad_differences - before_differences));
}

/* The extents of the shape sweep: zero, one, small, a vector's width on each side, and 127,
 * 128 and 129 on each side of the small-extent gate. */
static const int32_t ad_extents[] = {0, 1, 2, 3, 4, 5, 7, 8, 127, 128, 129};
#define AD_EXTENTS ((int)(sizeof(ad_extents) / sizeof(ad_extents[0])))

/* Every shape over ad_extents whose element count is at most AD_CAP, every axis, every
 * entry; with at most 16 elements every pattern, otherwise random bits and one other pattern
 * in turn. */
static void ad_shapes(void)
{
    const unsigned long before_cases = ad_cases;
    const unsigned long before_differences = ad_differences;
    uint32_t state = UINT32_C(0x2545f491);
    int turn = 0;
    for (int a = 0; a < AD_EXTENTS; ++a)
        for (int b = 0; b < AD_EXTENTS; ++b)
            for (int c = 0; c < AD_EXTENTS; ++c)
                for (int d = 0; d < AD_EXTENTS; ++d)
                {
                    const cmsis_nn_dims dims = {ad_extents[a], ad_extents[b], ad_extents[c], ad_extents[d]};
                    const long count = ad_small_count(&dims);
                    if (count < 0)
                        continue;
                    for (int entry = 0; entry < AD_ENTRIES; ++entry)
                        for (int32_t axis = 0; axis < 4; ++axis)
                        {
                            if (count <= 16)
                            {
                                for (int pattern = 0; pattern < AD_PATTERNS; ++pattern)
                                {
                                    ad_fill(pattern, ad_width(entry), (size_t)count, &state);
                                    ad_case(ad_pattern_names[pattern], entry, &dims, axis, 1, 1);
                                }
                                continue;
                            }
                            ad_fill(0, ad_width(entry), (size_t)count, &state);
                            ad_case(ad_pattern_names[0], entry, &dims, axis, 1, 1);
                            const int pattern = 1 + turn++ % (AD_PATTERNS - 1);
                            ad_fill(pattern, ad_width(entry), (size_t)count, &state);
                            ad_case(ad_pattern_names[pattern], entry, &dims, axis, 1, 1);
                        }
                }
    ad_report("shapes", before_cases, before_differences);
}

/* A NaN, and separately a zero of each sign, at every position of a short line, around a
 * winner at every other position: the first-extremum and NaN rules at each lane. */
static void ad_positions(void)
{
    const unsigned long before_cases = ad_cases;
    const unsigned long before_differences = ad_differences;
    for (int entry = 0; entry < AD_ENTRIES; ++entry)
    {
        const uint32_t *special = ad_width(entry) == sizeof(float32_t) ? ad_special32 : ad_special16;
        for (int32_t length = 1; length <= 9; ++length)
        {
            const cmsis_nn_dims dims = {1, 1, 1, length};
            for (int32_t odd = 0; odd < length; ++odd)
                for (int32_t winner = 0; winner < length; ++winner)
                    for (int kind = 0; kind < AD_SPECIALS; ++kind)
                    {
                        for (int32_t i = 0; i < length; ++i)
                            ad_bits[i] = special[11];
                        ad_bits[winner] = special[10];
                        ad_bits[odd] = special[kind];
                        ad_case("positions", entry, &dims, 3, 1, 1);
                    }
        }
    }
    ad_report("positions", before_cases, before_differences);
}

/* Every refusal of the previous body, and its empty successes: no shape, axes out of range,
 * a negative extent at each position (with each axis, and beside a zero extent), a zero
 * reduced extent, a missing input or output beside an empty and a nonempty shape, and
 * extents whose products pass INT32_MAX in bytes, or wrap 64 bits, at each axis. */
static void ad_refusals(void)
{
    const unsigned long before_cases = ad_cases;
    const unsigned long before_differences = ad_differences;
    uint32_t state = UINT32_C(0x9e3779b9);
    ad_fill(0, sizeof(float32_t), AD_CAP, &state);
    const int32_t big = INT32_C(1) << 30;
    const cmsis_nn_dims valid = {1, 2, 3, 4};
    const int32_t axes[] = {-1, 4, 5, INT32_MIN, INT32_MAX, 0, 3};
    const int32_t negatives[] = {-1, -2, -128, INT32_MIN};
    const cmsis_nn_dims overflowing[] = {{1, big, big, 16},
                                         {big, 1, big, 16},
                                         {big, big, 1, 16},
                                         {big, big, 16, 1},
                                         {big, big, 4, 4},
                                         {46341, 46341, 1, 1},
                                         {1, 1, 46341, 46341},
                                         {46341, 1, 1, 46341},
                                         {1, 1, 1, INT32_MAX},
                                         {1, 1, 536870912, 1},
                                         {1, 1, 1, 1073741824},
                                         {2, 1, 1, 1073741824},
                                         {128, 128, 128, 1024},
                                         {128, 128, 128, 256},
                                         {255, 255, 255, 255},
                                         {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX},
                                         {65536, 65536, 1, 1},
                                         {1, 65536, 32768, 1},
                                         {1, INT32_MAX, INT32_MAX, 0},
                                         {0, INT32_MAX, INT32_MAX, 1},
                                         {INT32_MAX, 0, 0, INT32_MAX}};
    for (int entry = 0; entry < AD_ENTRIES; ++entry)
    {
        for (int pass = 0; pass < 4; ++pass)
            ad_case("no dims", entry, NULL, 0, pass & 1, pass >> 1);
        for (unsigned i = 0; i < sizeof(axes) / sizeof(axes[0]); ++i)
            for (int pass = 0; pass < 4; ++pass)
                ad_case("axis", entry, &valid, axes[i], pass & 1, pass >> 1);
        for (int position = 0; position < 4; ++position)
            for (unsigned i = 0; i < sizeof(negatives) / sizeof(negatives[0]); ++i)
                for (int other = 0; other < 3; ++other)
                    for (int32_t axis = 0; axis < 4; ++axis)
                    {
                        int32_t d[4] = {2, 3, 1, 4};
                        if (other == 1)
                            d[(position + 1) & 3] = 0;
                        if (other == 2)
                            d[(position + 1) & 3] = INT32_MAX;
                        d[position] = negatives[i];
                        const cmsis_nn_dims dims = {d[0], d[1], d[2], d[3]};
                        ad_case("negative extent", entry, &dims, axis, 1, 1);
                    }
        for (int pass = 0; pass < 4; ++pass)
            for (int32_t axis = 0; axis < 4; ++axis)
            {
                const cmsis_nn_dims nonempty = {2, 1, 3, 2};
                const cmsis_nn_dims empty_line = {2, 0, 3, 2};
                const cmsis_nn_dims zero_reduced = {0, 1, 3, 2};
                ad_case("pointers nonempty", entry, &nonempty, axis, pass & 1, pass >> 1);
                ad_case("pointers empty", entry, &empty_line, axis, pass & 1, pass >> 1);
                ad_case("pointers zero reduced", entry, &zero_reduced, axis, pass & 1, pass >> 1);
            }
        for (unsigned i = 0; i < sizeof(overflowing) / sizeof(overflowing[0]); ++i)
            for (int32_t axis = 0; axis < 4; ++axis)
                for (int pass = 0; pass < 4; ++pass)
                    ad_case("overflowing", entry, &overflowing[i], axis, pass & 1, pass >> 1);
    }
    ad_report("refusals", before_cases, before_differences);
    snprintf(ad_message,
             sizeof(ad_message),
             "differential refusals: %lu accepted shapes beyond the buffers left to the decision cases",
             ad_decided_only);
    TEST_MESSAGE(ad_message);
}

/* The decision cases: the library's argument checks (arm_nn_arg_extrema_lines) against the
 * previous chain (ap_arg_lines) on shapes no buffer can hold, so both sides of the
 * small-extent gate and of every refusal threshold of the chain are reached without running
 * a line: extents of 127 and 128 (the gate), 46340 and 46341 (a pair above INT32_MAX),
 * 2^29 - 1 and 2^29, 2^30 - 1 and 2^30 (the byte bounds at four and two bytes), INT32_MAX,
 * negative extents, and pairs and triples of powers of two and their neighbours whose
 * products cross 2^29, 2^30, 2^31 and wrap 64 bits. A decision differs when the statuses
 * differ, or when both accept and the output counts differ, or, for a nonempty output, the
 * outer, inner or reduced counts. */
static unsigned long ad_decisions;
static unsigned long ad_decision_differences;
static unsigned long ad_sides[2][2];

static void ad_decide(const int32_t d[4], int32_t axis, size_t width)
{
    const cmsis_nn_dims dims = {d[0], d[1], d[2], d[3]};
    size_t old_outer = 0, old_inner = 0, old_reduction = 0;
    uint64_t old_lines = 0;
    size_t new_outer = 0, new_inner = 0, new_reduction = 0;
    const arm_cmsis_nn_status want =
        ap_arg_lines(&dims, axis, width, &old_outer, &old_inner, &old_reduction, &old_lines);
    const arm_cmsis_nn_status got =
        arm_nn_arg_extrema_lines(&dims, axis, width, &new_outer, &new_inner, &new_reduction);
    int differs = got != want;
    if (!differs && got == ARM_CMSIS_NN_SUCCESS)
    {
        differs |= (uint64_t)new_outer * (uint64_t)new_inner != old_lines;
        if (old_lines != 0)
            differs |= new_outer != old_outer || new_inner != old_inner || new_reduction != old_reduction;
    }
    const int gated = (d[0] | d[1] | d[2] | d[3]) >= 0 && (d[0] | d[1] | d[2] | d[3]) < 128;
    ++ad_sides[gated][want == ARM_CMSIS_NN_SUCCESS];
    ++ad_decisions;
    if (differs)
    {
        if (ad_decision_differences < AD_REPORTED)
        {
            snprintf(ad_message,
                     sizeof(ad_message),
                     "decision difference: dims {%ld, %ld, %ld, %ld} axis %ld width %u status %d against %d",
                     (long)d[0],
                     (long)d[1],
                     (long)d[2],
                     (long)d[3],
                     (long)axis,
                     (unsigned)width,
                     (int)got,
                     (int)want);
            TEST_MESSAGE(ad_message);
        }
        ++ad_decision_differences;
    }
}

static const int32_t ad_thresholds[] = {-1,
                                        INT32_MIN,
                                        0,
                                        1,
                                        2,
                                        3,
                                        126,
                                        127,
                                        128,
                                        129,
                                        255,
                                        256,
                                        46340,
                                        46341,
                                        INT32_C(536870911),
                                        INT32_C(536870912),
                                        INT32_C(1073741823),
                                        INT32_C(1073741824),
                                        INT32_MAX};
#define AD_THRESHOLDS ((int)(sizeof(ad_thresholds) / sizeof(ad_thresholds[0])))

static void ad_decide_all(const int32_t d[4], int32_t first_axis, int32_t last_axis)
{
    for (int32_t axis = first_axis; axis <= last_axis; ++axis)
    {
        ad_decide(d, axis, sizeof(float32_t));
        ad_decide(d, axis, sizeof(uint16_t));
    }
}

static void ad_decisions_sweep(void)
{
    /* Every shape over the thresholds, at axes -1 to 4. */
    for (int a = 0; a < AD_THRESHOLDS; ++a)
        for (int b = 0; b < AD_THRESHOLDS; ++b)
            for (int c = 0; c < AD_THRESHOLDS; ++c)
                for (int e = 0; e < AD_THRESHOLDS; ++e)
                {
                    const int32_t d[4] = {ad_thresholds[a], ad_thresholds[b], ad_thresholds[c], ad_thresholds[e]};
                    ad_decide_all(d, -1, 4);
                }
    /* Two extents over 2^k - 1, 2^k and 2^k + 1 (k = 0 to 30) and INT32_MAX, the others 1. */
    int32_t powers[94];
    int count = 0;
    for (int k = 0; k <= 30; ++k)
    {
        const int32_t p = INT32_C(1) << k;
        powers[count++] = p - 1;
        powers[count++] = p;
        powers[count++] = p + 1;
    }
    powers[count++] = INT32_MAX;
    for (int first = 0; first < 4; ++first)
        for (int second = first + 1; second < 4; ++second)
            for (int i = 0; i < count; ++i)
                for (int j = 0; j < count; ++j)
                {
                    int32_t d[4] = {1, 1, 1, 1};
                    d[first] = powers[i];
                    d[second] = powers[j];
                    ad_decide_all(d, 0, 3);
                }
    /* Three extents over powers of two and their lower neighbours, the fourth 1. */
    const int32_t triples[] = {1,
                               2,
                               31,
                               32,
                               511,
                               512,
                               1023,
                               1024,
                               1025,
                               16383,
                               16384,
                               32768,
                               65536,
                               524288,
                               1048576,
                               INT32_C(536870911),
                               INT32_C(536870912),
                               INT32_C(1073741824)};
    const int triple_count = (int)(sizeof(triples) / sizeof(triples[0]));
    for (int one = 0; one < 4; ++one)
        for (int i = 0; i < triple_count; ++i)
            for (int j = 0; j < triple_count; ++j)
                for (int k = 0; k < triple_count; ++k)
                {
                    int32_t d[4];
                    const int32_t picks[3] = {triples[i], triples[j], triples[k]};
                    for (int p = 0, q = 0; p < 4; ++p)
                        d[p] = p == one ? 1 : picks[q++];
                    ad_decide_all(d, 0, 3);
                }
    /* No shape at all. */
    size_t outer, inner, reduction;
    if (arm_nn_arg_extrema_lines(NULL, 0, sizeof(float32_t), &outer, &inner, &reduction) != ARM_CMSIS_NN_ARG_ERROR)
        ++ad_decision_differences;
    ++ad_decisions;
    snprintf(ad_message,
             sizeof(ad_message),
             "differential decisions: %lu cases, %lu differences; inside the gate %lu accepted %lu refused, outside "
             "%lu accepted %lu refused",
             ad_decisions,
             ad_decision_differences,
             ad_sides[1][1],
             ad_sides[1][0],
             ad_sides[0][1],
             ad_sides[0][0]);
    TEST_MESSAGE(ad_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)ad_decision_differences);
}
