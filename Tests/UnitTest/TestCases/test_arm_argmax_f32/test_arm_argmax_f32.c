/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#define AE_HALF 0
#define AE_MAX 1
#define AE_KERNEL arm_argmax_f32
#include "../Utils/arg_extrema_flt_test_common.h"

/* Extents whose 64-bit products wrap. Three int32 extents of 2^30, 2^30 and 16
 * multiply to 2^64, which a 64-bit product reads as 0, so a bound taken on the
 * wrapped product would admit the array as empty and return success without
 * writing. The array has 2^64 elements, beyond the INT32_MAX-byte bound, and is
 * refused like any other oversized array, with nothing written, as the count
 * checks of the base refused it. One shape per axis, each wrapping another
 * product: the extents after the axis (axis 0), the output count (axis 1) and
 * the extents before it (axes 2 and 3). */
static void ae_extent_wrap(void)
{
    const int32_t big = INT32_C(1) << 30;
    const cmsis_nn_dims shapes[4] = {{1, big, big, 16}, {big, 1, big, 16}, {big, big, 1, 16}, {big, big, 16, 1}};
    for (int32_t axis = 0; axis < 4; ++axis)
    {
        for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
            ae_output[i] = AE_SENTINEL;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &shapes[axis], axis, ae_output + AE_GUARD));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(NULL, &shapes[axis], axis, NULL));
        for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
            TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[i]);
    }
    /* Here the output count, 2^62, does not wrap, but the input's element count
     * (2^64) and the output's byte count (2^64) both do, so a wrapped bound
     * would admit 2^62 writes. Refused as well. (A case that can only fail by
     * writing out of bounds, so it follows the fix.) */
    const cmsis_nn_dims writes = {big, big, 4, 4};
    for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
        ae_output[i] = AE_SENTINEL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &writes, 3, ae_output + AE_GUARD));
    for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[i]);
    /* A zero extent still makes an empty array whatever the others are, with
     * the zero last as well as first: success, nothing read or written. */
    const cmsis_nn_dims zero_last = {1, INT32_MAX, INT32_MAX, 0};
    const cmsis_nn_dims zero_first = {0, INT32_MAX, INT32_MAX, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(NULL, &zero_last, 0, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(NULL, &zero_first, 3, NULL));
}

/* A long line: c = 4096 along axis 3, where the lines above stop at 256 (a kNN
 * reference axis runs into the hundreds and thousands). Each line is checked
 * against a plain loop over that line alone, which decodes every value with the
 * suite's binary64 decoder and applies the suite's rule: the first element
 * seeds, a NaN never replaces, a value replaces a NaN seed, and only a strictly
 * better value replaces another. The constructed lines also assert their index
 * outright. Lines: the suite's xorshift encodings; a NaN at the first, second,
 * last, and the winner's neighbouring positions; a NaN everywhere but one
 * position; a leading run of NaNs; +0/-0 ties as the extremum; an all-equal
 * tail; an all-equal line; and an all-NaN line. */
#define AE_LONG 4096
#define AE_LONG_LINES 2
#define AE_LONG_TWO_BITS (AE_ONE_BITS + (UINT32_C(1) << AE_FRAC))
#if AE_MAX
    #define AE_LONG_WEAK (AE_SIGN_MASK | AE_ONE_BITS)
    #define AE_LONG_STRONG AE_LONG_TWO_BITS
#else
    #define AE_LONG_WEAK AE_ONE_BITS
    #define AE_LONG_STRONG (AE_SIGN_MASK | AE_LONG_TWO_BITS)
#endif

static AE_TYPE ae_long_storage[AE_LONG_LINES * AE_LONG + 2 * AE_GUARD];
static AE_BITS ae_long_bits[AE_LONG_LINES * AE_LONG];
static int32_t ae_long_output[AE_LONG_LINES + 2 * AE_GUARD];

static int32_t ae_long_reference(const AE_BITS *line)
{
    int32_t best = 0;
    for (int32_t k = 1; k < AE_LONG; ++k)
    {
        if (ae_is_nan(line[k]))
            continue;
#if AE_MAX
        if (ae_is_nan(line[best]) || ae_value(line[k]) > ae_value(line[best]))
#else
        if (ae_is_nan(line[best]) || ae_value(line[k]) < ae_value(line[best]))
#endif
            best = k;
    }
    return best;
}

/* `expected` holds each line's index, or -1 where only the reference decides. */
static void ae_long_check(const int32_t expected[AE_LONG_LINES])
{
    const cmsis_nn_dims dims = {1, 1, AE_LONG_LINES, AE_LONG};
    memset(ae_long_storage, 0xa5, sizeof(ae_long_storage));
    memcpy(ae_long_storage + AE_GUARD, ae_long_bits, sizeof(ae_long_bits));
    for (int i = 0; i < AE_LONG_LINES + 2 * AE_GUARD; ++i)
        ae_long_output[i] = AE_SENTINEL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(ae_long_storage + AE_GUARD, &dims, 3, ae_long_output + AE_GUARD));
    for (int line = 0; line < AE_LONG_LINES; ++line)
    {
        const int32_t reference = ae_long_reference(ae_long_bits + line * AE_LONG);
        if (expected[line] >= 0)
            TEST_ASSERT_EQUAL_INT32(expected[line], reference);
        TEST_ASSERT_EQUAL_INT32(reference, ae_long_output[AE_GUARD + line]);
    }
    for (int i = 0; i < AE_GUARD; ++i)
    {
        TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_long_output[i]);
        TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_long_output[AE_GUARD + AE_LONG_LINES + i]);
    }
    TEST_ASSERT_EQUAL_MEMORY(ae_long_bits, ae_long_storage + AE_GUARD, sizeof(ae_long_bits));
}

static void ae_long_fill(AE_BITS *line, uint32_t bits)
{
    for (int k = 0; k < AE_LONG; ++k)
        line[k] = (AE_BITS)bits;
}

static void ae_long_line(void)
{
    const uint32_t nan = AE_EXPONENT_MASK | 1;
    const uint32_t quiet_negative_nan = AE_SIGN_MASK | AE_EXPONENT_MASK | ((AE_FRACTION_MASK >> 1) + 1);
    AE_BITS *first = ae_long_bits;
    AE_BITS *second = ae_long_bits + AE_LONG;
    const int32_t unknown[AE_LONG_LINES] = {-1, -1};

    /* The suite's xorshift encodings: every class of value, NaNs included. */
    uint32_t state = UINT32_C(0x9e3779b9);
    for (int run = 0; run < 8; ++run)
    {
        for (int i = 0; i < AE_LONG_LINES * AE_LONG; ++i)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ae_long_bits[i] = (AE_BITS)state;
        }
        ae_long_check(unknown);
    }

    /* A NaN at each position class around one winner at 2049. */
    const int32_t winner = 2049;
    const int32_t nan_positions[] = {0, 1, winner - 1, winner + 1, AE_LONG - 1};
    for (unsigned i = 0; i < sizeof(nan_positions) / sizeof(nan_positions[0]); ++i)
    {
        ae_long_fill(first, AE_LONG_WEAK);
        first[winner] = (AE_BITS)AE_LONG_STRONG;
        first[nan_positions[i]] = (AE_BITS)nan;
        ae_long_fill(second, AE_LONG_WEAK);
        second[nan_positions[i]] = (AE_BITS)quiet_negative_nan;
        const int32_t expected[AE_LONG_LINES] = {winner, nan_positions[i] == 0 ? 1 : 0};
        ae_long_check(expected);
    }

    /* A NaN everywhere but the last position, and everywhere but the first. */
    ae_long_fill(first, nan);
    first[AE_LONG - 1] = (AE_BITS)AE_LONG_WEAK;
    ae_long_fill(second, quiet_negative_nan);
    second[0] = (AE_BITS)AE_LONG_WEAK;
    {
        const int32_t expected[AE_LONG_LINES] = {AE_LONG - 1, 0};
        ae_long_check(expected);
    }

    /* A leading run of NaNs, then values with the winner at 3000. */
    ae_long_fill(first, AE_LONG_WEAK);
    for (int k = 0; k < 2048; ++k)
        first[k] = (AE_BITS)(nan | (AE_SIGN_MASK * (uint32_t)(k & 1)));
    first[3000] = (AE_BITS)AE_LONG_STRONG;
    /* +0/-0 ties as the extremum: the first zero wins, whatever its sign. */
    ae_long_fill(second, AE_LONG_WEAK);
    second[100] = (AE_BITS)AE_SIGN_MASK;
    second[200] = 0;
    second[AE_LONG - 1] = (AE_BITS)AE_SIGN_MASK;
    {
        const int32_t expected[AE_LONG_LINES] = {3000, 100};
        ae_long_check(expected);
    }

    /* Zeros of alternating sign across the whole line; and an all-equal tail
     * holding the extremum from 2048 on. */
    for (int k = 0; k < AE_LONG; ++k)
        first[k] = (AE_BITS)(AE_SIGN_MASK * (uint32_t)(k & 1));
    ae_long_fill(second, AE_LONG_WEAK);
    for (int k = 2048; k < AE_LONG; ++k)
        second[k] = (AE_BITS)AE_LONG_STRONG;
    {
        const int32_t expected[AE_LONG_LINES] = {0, 2048};
        ae_long_check(expected);
    }

    /* An all-equal tail whose value already appeared at 7; an all-equal line. */
    ae_long_fill(first, AE_LONG_WEAK);
    first[7] = (AE_BITS)AE_LONG_STRONG;
    for (int k = 2048; k < AE_LONG; ++k)
        first[k] = (AE_BITS)AE_LONG_STRONG;
    ae_long_fill(second, AE_LONG_STRONG);
    {
        const int32_t expected[AE_LONG_LINES] = {7, 0};
        ae_long_check(expected);
    }

    /* All-NaN lines, of one payload and of mixed payloads and signs. */
    ae_long_fill(first, nan);
    for (int k = 0; k < AE_LONG; ++k)
        second[k] =
            (AE_BITS)(AE_EXPONENT_MASK | (AE_SIGN_MASK * (uint32_t)(k & 1)) | (1 + (uint32_t)k % AE_FRACTION_MASK));
    {
        const int32_t expected[AE_LONG_LINES] = {0, 0};
        ae_long_check(expected);
    }
}
