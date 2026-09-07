/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Shared body of the test_arm_{split,unpack,concatenation,pack}_{f32,f16} suites (#411).
 *
 * The including suite defines, before this header:
 *   AC_OP      AC_OP_SPLIT, AC_OP_UNPACK, AC_OP_CONCAT or AC_OP_PACK
 *   AC_F16     0 for float32_t, 1 for float16_t
 *   AC_PREFIX  token prefix of the suite's test functions, e.g. split_f32
 *   AC_KERNEL  the kernel under test
 *
 * Every op is a bit copy between one "packed" tensor and a set of "slices" along one axis, so one
 * NumPy-semantics reference serves all four: ac_ref_map() decomposes a packed row-major flat index into
 * a multi-index, finds the slice that owns its axis coordinate, and re-flattens into that slice. Nothing
 * is derived from the kernel's own outer/inner decomposition. Inputs are random bit patterns with NaN
 * (quiet and signaling, with payloads), +/-Inf, +/-0 and subnormals injected, and every comparison is
 * bit-for-bit. Every output sits between guard words that must survive, and the argument-error cases
 * require the outputs to be untouched.
 */

#pragma once

#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#define AC_OP_SPLIT 0
#define AC_OP_UNPACK 1
#define AC_OP_CONCAT 2
#define AC_OP_PACK 3

/* Scatter: packed is the kernel input, slices its outputs. Gather: the reverse. */
#define AC_SCATTER (AC_OP == AC_OP_SPLIT || AC_OP == AC_OP_UNPACK)
/* Unit: every slice is one element wide along the axis, and the axis is absent from the slice shape.
 * Pack alone takes the slice shape (the axis is inserted); unpack takes the packed shape. */
#define AC_UNIT (AC_OP == AC_OP_UNPACK || AC_OP == AC_OP_PACK)

#define AC_CAT_(a, b) a##_##b
#define AC_CAT(a, b) AC_CAT_(a, b)
#define AC_FN(name) AC_CAT(AC_PREFIX, name)

#if AC_F16
typedef float16_t ac_t;
typedef uint16_t ac_bits_t;
    #define AC_ASSERT_BITS(e, a) TEST_ASSERT_EQUAL_HEX16((e), (a))
    #define AC_EXP_MASK 0x7C00u
    #define AC_MANT_MASK 0x03FFu
    #define AC_SIGN 0x8000u
    #define AC_QUIET 0x0200u
    #define AC_LANES 8
#else
typedef float32_t ac_t;
typedef uint32_t ac_bits_t;
    #define AC_ASSERT_BITS(e, a) TEST_ASSERT_EQUAL_HEX32((e), (a))
    #define AC_EXP_MASK 0x7F800000u
    #define AC_MANT_MASK 0x007FFFFFu
    #define AC_SIGN 0x80000000u
    #define AC_QUIET 0x00400000u
    #define AC_LANES 4
#endif

#define AC_MAX_DIMS 6
#define AC_MAX_SLICES 16
#define AC_GUARD 8
#define AC_MAX_ELEMENTS 4096
#define AC_ARENA (AC_MAX_ELEMENTS + 2 * AC_GUARD * AC_MAX_SLICES)
#ifndef AC_FUZZ_ITERS
    #define AC_FUZZ_ITERS 400
#endif

static ac_t ac_packed_buf[AC_MAX_ELEMENTS + 2 * AC_GUARD];
static ac_t ac_slice_arena[AC_ARENA];
static ac_t *ac_slices[AC_MAX_SLICES];
static int32_t ac_slice_len[AC_MAX_SLICES];

/* ---------------------------------------------------------------- helpers */

static uint32_t ac_seed;
static uint32_t ac_rand(void)
{
    ac_seed = ac_seed * 1664525u + 1013904223u;
    return ac_seed;
}
static int32_t ac_rand_range(int32_t lo, int32_t hi) { return lo + (int32_t)(ac_rand() % (uint32_t)(hi - lo + 1)); }

static ac_bits_t ac_bits_of(ac_t x)
{
    ac_bits_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}
static ac_t ac_of_bits(ac_bits_t b)
{
    ac_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

/* One in eight elements is a special: NaN with payload (quiet or signaling), +/-Inf, +/-0 or subnormal. */
static ac_bits_t ac_random_bits(void)
{
    const uint32_t r = ac_rand();
    const ac_bits_t sign = (r & 1u) ? (ac_bits_t)AC_SIGN : 0u;
    const ac_bits_t payload = (ac_bits_t)(AC_MANT_MASK & (ac_rand() | 1u));
    if ((r & 0x70u) != 0x70u)
    {
        return (ac_bits_t)(ac_rand() >> (32 - 8 * sizeof(ac_bits_t)));
    }
    switch ((r >> 8) & 7u)
    {
    case 0:
        return (ac_bits_t)(sign | AC_EXP_MASK | AC_QUIET | payload); /* qNaN, payload */
    case 1:
        return (ac_bits_t)(sign | AC_EXP_MASK | (payload & (ac_bits_t)~AC_QUIET)); /* sNaN, payload */
    case 2:
        return (ac_bits_t)(sign | AC_EXP_MASK); /* Inf */
    case 3:
        return sign; /* zero */
    default:
        return (ac_bits_t)(sign | payload); /* subnormal */
    }
}

static void ac_fill_random(ac_t *buf, int32_t n)
{
    for (int32_t i = 0; i < n; i++)
    {
        buf[i] = ac_of_bits(ac_random_bits());
    }
}

#define AC_GUARD_BITS ((ac_bits_t)(0xA5A5A5A5u & (ac_bits_t)~0u))

static void ac_fill_guard(ac_t *buf, int32_t n)
{
    for (int32_t i = 0; i < n; i++)
    {
        buf[i] = ac_of_bits(AC_GUARD_BITS);
    }
}

static void ac_assert_guard(const ac_t *buf, int32_t n)
{
    for (int32_t i = 0; i < n; i++)
    {
        AC_ASSERT_BITS(AC_GUARD_BITS, ac_bits_of(buf[i]));
    }
}

/* Lays out num slices of the given lengths in the arena, each between AC_GUARD guard words. */
static void ac_layout_slices(int32_t num, const int32_t *lens)
{
    ac_t *p = ac_slice_arena;
    TEST_ASSERT_TRUE(num <= AC_MAX_SLICES);
    for (int32_t s = 0; s < num; s++)
    {
        TEST_ASSERT_TRUE((p + AC_GUARD + lens[s] + AC_GUARD) <= (ac_slice_arena + AC_ARENA));
        ac_fill_guard(p, AC_GUARD);
        ac_slices[s] = p + AC_GUARD;
        ac_slice_len[s] = lens[s];
        ac_fill_guard(ac_slices[s] + lens[s], AC_GUARD);
        p += AC_GUARD + lens[s] + AC_GUARD;
    }
}

static void ac_assert_slice_guards(int32_t num)
{
    for (int32_t s = 0; s < num; s++)
    {
        ac_assert_guard(ac_slices[s] - AC_GUARD, AC_GUARD);
        ac_assert_guard(ac_slices[s] + ac_slice_len[s], AC_GUARD);
    }
}

static int32_t ac_product(const int32_t *shape, int32_t dims)
{
    int32_t p = 1;
    for (int32_t d = 0; d < dims; d++)
    {
        p *= shape[d];
    }
    return p;
}

/*
 * One packed case: `shape` is the full (packed) row-major shape of `dims` dimensions with `axis`
 * present; slice s spans sizes[s] along the axis (all 1 for the unit ops). The kernel-facing shape of
 * the unit ops is `shape` with the axis removed.
 */
typedef struct
{
    int32_t dims;
    int32_t shape[AC_MAX_DIMS];
    int32_t axis;
    int32_t num;
    int32_t sizes[AC_MAX_SLICES];
} ac_case_t;

/* NumPy semantics: packed flat index -> (slice, flat index within that slice). */
static void ac_ref_map(const ac_case_t *c, int32_t flat, int32_t *slice, int32_t *slice_flat)
{
    int32_t idx[AC_MAX_DIMS] = {0};
    for (int32_t d = c->dims - 1; d >= 0; d--)
    {
        idx[d] = flat % c->shape[d];
        flat /= c->shape[d];
    }
    int32_t a = idx[c->axis];
    int32_t s = 0;
    while (a >= c->sizes[s])
    {
        a -= c->sizes[s];
        s++;
    }
    idx[c->axis] = a;
    int32_t f = 0;
    for (int32_t d = 0; d < c->dims; d++)
    {
        const int32_t extent = (d == c->axis) ? c->sizes[s] : c->shape[d];
        f = f * extent + idx[d];
    }
    *slice = s;
    *slice_flat = f;
}

static int32_t ac_slice_elems(const ac_case_t *c, int32_t s)
{
    int32_t n = 1;
    for (int32_t d = 0; d < c->dims; d++)
    {
        n *= (d == c->axis) ? c->sizes[s] : c->shape[d];
    }
    return n;
}

/* Runs the kernel on a fresh random instance of `c` and checks every element and guard. */
static arm_cmsis_nn_status ac_call(const ac_case_t *c, ac_t *packed, ac_t *const *slices)
{
    int32_t kshape[AC_MAX_DIMS];
    int32_t kdims = 0;
    for (int32_t d = 0; d < c->dims; d++)
    {
        if (AC_OP == AC_OP_PACK && d == c->axis)
        {
            continue;
        }
        kshape[kdims++] = c->shape[d];
    }
#if AC_OP == AC_OP_SPLIT
    return AC_KERNEL(packed, kdims, kshape, c->axis, c->num, c->sizes, slices);
#elif AC_OP == AC_OP_UNPACK
    return AC_KERNEL(packed, kdims, kshape, c->axis, slices);
#elif AC_OP == AC_OP_CONCAT
    return AC_KERNEL((const ac_t *const *)slices, c->num, c->sizes, kdims, kshape, c->axis, packed);
#else
    return AC_KERNEL((const ac_t *const *)slices, c->num, kdims, kshape, c->axis, packed);
#endif
}

static void ac_run_case(const ac_case_t *c)
{
    const int32_t total = ac_product(c->shape, c->dims);
    int32_t lens[AC_MAX_SLICES];
    TEST_ASSERT_TRUE(total <= AC_MAX_ELEMENTS);
    TEST_ASSERT_TRUE(c->num >= 1 && c->num <= AC_MAX_SLICES);
    for (int32_t s = 0; s < c->num; s++)
    {
        lens[s] = ac_slice_elems(c, s);
    }
    ac_layout_slices(c->num, lens);
    ac_t *packed = ac_packed_buf + AC_GUARD;
    ac_fill_guard(ac_packed_buf, AC_GUARD);
    ac_fill_guard(packed + total, AC_GUARD);
#if AC_SCATTER
    ac_fill_random(packed, total);
    for (int32_t s = 0; s < c->num; s++)
    {
        ac_fill_guard(ac_slices[s], lens[s]);
    }
#else
    for (int32_t s = 0; s < c->num; s++)
    {
        ac_fill_random(ac_slices[s], lens[s]);
    }
    ac_fill_guard(packed, total);
#endif
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, ac_call(c, packed, ac_slices));
    for (int32_t i = 0; i < total; i++)
    {
        int32_t s;
        int32_t j;
        ac_ref_map(c, i, &s, &j);
        TEST_ASSERT_TRUE(s < c->num && j < lens[s]);
        AC_ASSERT_BITS(ac_bits_of(packed[i]), ac_bits_of(ac_slices[s][j]));
    }
    ac_assert_guard(ac_packed_buf, AC_GUARD);
    ac_assert_guard(packed + total, AC_GUARD);
    ac_assert_slice_guards(c->num);
}

#if !AC_UNIT
/* Random composition of `extent` into `num` non-negative parts (unequal on purpose). */
static void ac_random_sizes(ac_case_t *c, int32_t extent, int32_t num, bool allow_zero)
{
    c->num = num;
    int32_t left = extent;
    for (int32_t s = 0; s < num; s++)
    {
        const int32_t remaining_slots = num - s - 1;
        const int32_t min_here = allow_zero ? 0 : 1;
        const int32_t max_here = left - remaining_slots * min_here;
        c->sizes[s] = (s == num - 1) ? left : ac_rand_range(min_here, max_here);
        left -= c->sizes[s];
    }
}
#endif

/* Random case: rank, shape, axis, and slices; total elements bounded by max_total. */
static void ac_random_case(ac_case_t *c, int32_t dims, int32_t axis, int32_t max_total)
{
    c->dims = dims;
    c->axis = axis;
    for (;;)
    {
        for (int32_t d = 0; d < dims; d++)
        {
            c->shape[d] = ac_rand_range(1, 9);
        }
#if AC_UNIT
        c->shape[axis] = ac_rand_range(1, 8);
#else
        c->shape[axis] = ac_rand_range(1, 16);
#endif
        if (ac_product(c->shape, dims) <= max_total)
        {
            break;
        }
    }
#if AC_UNIT
    c->num = c->shape[axis];
    for (int32_t s = 0; s < c->num; s++)
    {
        c->sizes[s] = 1;
    }
#else
    ac_random_sizes(c, c->shape[axis], ac_rand_range(1, c->shape[axis] < 8 ? c->shape[axis] : 8), false);
#endif
}

/* ---------------------------------------------------------------- cases */

/* Rank 1..5 with the axis at every position, unequal slices, random shapes. */
void AC_FN(rank_axis_sweep)(void)
{
    ac_seed = 0x1234u + AC_OP;
    for (int32_t dims = 1; dims <= 5; dims++)
    {
        for (int32_t axis = 0; axis < dims; axis++)
        {
            for (int32_t rep = 0; rep < 3; rep++)
            {
                ac_case_t c;
                ac_random_case(&c, dims, axis, AC_MAX_ELEMENTS);
                ac_run_case(&c);
            }
        }
    }
}

/* Inner runs that are not vector multiples: 1, 3, 5, 7, 9, 13, 17, 31, 33 elements per copy. */
void AC_FN(tail_lengths)(void)
{
    static const int32_t inner[] = {1, 3, 5, 7, 9, 13, 17, 31, 33};
    ac_seed = 0x5151u + AC_OP;
    for (size_t k = 0; k < sizeof(inner) / sizeof(inner[0]); k++)
    {
        ac_case_t c;
        c.dims = 3;
        c.shape[0] = 3;
        c.shape[1] = AC_UNIT ? 5 : 7;
        c.shape[2] = inner[k];
        c.axis = 1;
#if AC_UNIT
        c.num = 5;
        for (int32_t s = 0; s < 5; s++)
        {
            c.sizes[s] = 1;
        }
#else
        c.num = 3;
        c.sizes[0] = 1;
        c.sizes[1] = 4;
        c.sizes[2] = 2;
#endif
        ac_run_case(&c);
        /* Axis last: the copy run is a single element wide per slice, the worst case for a memcpy. */
        c.axis = 2;
        c.shape[1] = inner[k];
        c.shape[2] = AC_UNIT ? 5 : 7;
        ac_run_case(&c);
    }
}

#if AC_UNIT
/* N = 1..8 slices at rank 1..4, plus the rank-0 stack (AC_OP_PACK only). */
void AC_FN(n_up_to_8)(void)
{
    ac_seed = 0x77u + AC_OP;
    for (int32_t n = 1; n <= 8; n++)
    {
        for (int32_t dims = 1; dims <= 4; dims++)
        {
            ac_case_t c;
            c.dims = dims;
            c.axis = ac_rand_range(0, dims - 1);
            for (int32_t d = 0; d < dims; d++)
            {
                c.shape[d] = ac_rand_range(1, 6);
            }
            c.shape[c.axis] = n;
            c.num = n;
            for (int32_t s = 0; s < n; s++)
            {
                c.sizes[s] = 1;
            }
            ac_run_case(&c);
        }
    }
    #if AC_OP == AC_OP_PACK
    {
        /* Rank-0 inputs: pack of N scalars is a vector of N. Exercised directly. */
        ac_t in[8];
        const ac_t *ins[8];
        ac_t out[8 + 2 * AC_GUARD];
        ac_fill_random(in, 8);
        for (int32_t s = 0; s < 8; s++)
        {
            ins[s] = &in[s];
        }
        ac_fill_guard(out, 8 + 2 * AC_GUARD);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AC_KERNEL(ins, 8, 0, NULL, 0, out + AC_GUARD));
        for (int32_t s = 0; s < 8; s++)
        {
            AC_ASSERT_BITS(ac_bits_of(in[s]), ac_bits_of(out[AC_GUARD + s]));
        }
        ac_assert_guard(out, AC_GUARD);
        ac_assert_guard(out + AC_GUARD + 8, AC_GUARD);
    }
    #endif
}
#else
/* Unequal slice lengths (SPLIT_V shapes), including a zero-length slice. */
void AC_FN(unequal_sizes)(void)
{
    ac_seed = 0x99u + AC_OP;
    {
        ac_case_t c = {3, {2, 7, 3}, 1, 3, {1, 4, 2}};
        ac_run_case(&c);
    }
    {
        ac_case_t c = {1, {13}, 0, 2, {5, 8}};
        ac_run_case(&c);
    }
    {
        ac_case_t c = {4, {2, 3, 10, 5}, 2, 4, {3, 0, 6, 1}};
        ac_run_case(&c);
    }
    {
        ac_case_t c = {5, {2, 2, 9, 3, 2}, 2, 5, {1, 1, 5, 1, 1}};
        ac_run_case(&c);
    }
    {
        ac_case_t c = {2, {16, 2}, 0, 16, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
        ac_run_case(&c);
    }
}
#endif

/* Every element a special value: NaN payloads (quiet and signaling), +/-Inf, +/-0, subnormals. */
void AC_FN(payload_preservation)(void)
{
    ac_case_t c = {3, {3, AC_UNIT ? 4 : 6, 11}, 1, 0, {0}};
    static const ac_bits_t specials[] = {
        (ac_bits_t)(AC_EXP_MASK | AC_QUIET | (AC_MANT_MASK & 0x2AAAAu)),           /* qNaN, payload */
        (ac_bits_t)(AC_SIGN | AC_EXP_MASK | AC_QUIET | (AC_MANT_MASK & 0x15555u)), /* -qNaN, payload */
        (ac_bits_t)(AC_EXP_MASK | 1u),                                             /* sNaN */
        (ac_bits_t)(AC_SIGN | AC_EXP_MASK | ((AC_MANT_MASK >> 1) & 0x3F3u)),       /* -sNaN, payload */
        (ac_bits_t)AC_EXP_MASK,                                         /* +Inf */
        (ac_bits_t)(AC_SIGN | AC_EXP_MASK),                             /* -Inf */
        0u,                                                             /* +0 */
        (ac_bits_t)AC_SIGN,                                             /* -0 */
        1u,                                                             /* smallest subnormal */
        (ac_bits_t)(AC_SIGN | AC_MANT_MASK),                            /* -largest subnormal */
        (ac_bits_t)(AC_EXP_MASK - 1u),                                  /* max finite */
    };
    const int32_t total = ac_product(c.shape, c.dims);
    int32_t lens[AC_MAX_SLICES];
#if AC_UNIT
    c.num = c.shape[1];
    for (int32_t s = 0; s < c.num; s++)
    {
        c.sizes[s] = 1;
    }
#else
    c.num = 3;
    c.sizes[0] = 2;
    c.sizes[1] = 1;
    c.sizes[2] = 3;
#endif
    for (int32_t s = 0; s < c.num; s++)
    {
        lens[s] = ac_slice_elems(&c, s);
    }
    ac_layout_slices(c.num, lens);
    ac_t *packed = ac_packed_buf + AC_GUARD;
    ac_fill_guard(ac_packed_buf, AC_GUARD);
    ac_fill_guard(packed + total, AC_GUARD);
    const size_t nspec = sizeof(specials) / sizeof(specials[0]);
#if AC_SCATTER
    for (int32_t i = 0; i < total; i++)
    {
        packed[i] = ac_of_bits(specials[(size_t)i % nspec]);
    }
    for (int32_t s = 0; s < c.num; s++)
    {
        ac_fill_guard(ac_slices[s], lens[s]);
    }
#else
    for (int32_t s = 0; s < c.num; s++)
    {
        for (int32_t j = 0; j < lens[s]; j++)
        {
            ac_slices[s][j] = ac_of_bits(specials[(size_t)(j + s) % nspec]);
        }
    }
    ac_fill_guard(packed, total);
#endif
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, ac_call(&c, packed, ac_slices));
    for (int32_t i = 0; i < total; i++)
    {
        int32_t s;
        int32_t j;
        ac_ref_map(&c, i, &s, &j);
        AC_ASSERT_BITS(ac_bits_of(packed[i]), ac_bits_of(ac_slices[s][j]));
    }
    ac_assert_guard(ac_packed_buf, AC_GUARD);
    ac_assert_guard(packed + total, AC_GUARD);
    ac_assert_slice_guards(c.num);
}

/* A zero-extent dimension: success, nothing written. */
void AC_FN(zero_extent)(void)
{
    ac_case_t c = {3, {2, AC_UNIT ? 3 : 4, 0}, 1, 0, {0}};
#if AC_UNIT
    c.num = 3;
    c.sizes[0] = c.sizes[1] = c.sizes[2] = 1;
#else
    c.num = 2;
    c.sizes[0] = 1;
    c.sizes[1] = 3;
#endif
    ac_run_case(&c);
#if !AC_UNIT
    /* Zero along the axis itself, with every slice empty. */
    c.shape[1] = 0;
    c.shape[2] = 5;
    c.sizes[0] = 0;
    c.sizes[1] = 0;
    ac_run_case(&c);
#endif
}

/* Invalid arguments return ARG_ERROR and leave every output word untouched. */
void AC_FN(arg_error)(void)
{
    ac_case_t good = {3, {2, AC_UNIT ? 3 : 6, 5}, 1, 0, {0}};
    int32_t lens[AC_MAX_SLICES];
#if AC_UNIT
    good.num = 3;
    good.sizes[0] = good.sizes[1] = good.sizes[2] = 1;
#else
    good.num = 3;
    good.sizes[0] = 1;
    good.sizes[1] = 2;
    good.sizes[2] = 3;
#endif
    const int32_t total = ac_product(good.shape, good.dims);
    for (int32_t s = 0; s < good.num; s++)
    {
        lens[s] = ac_slice_elems(&good, s);
    }
    ac_layout_slices(good.num, lens);
    ac_t *packed = ac_packed_buf + AC_GUARD;

    /* The whole output side is guard pattern; the input side is random. */
#define AC_PREP()                                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        ac_fill_guard(ac_packed_buf, total + 2 * AC_GUARD);                                                            \
        for (int32_t s_ = 0; s_ < good.num; s_++)                                                                      \
        {                                                                                                              \
            ac_fill_guard(ac_slices[s_], lens[s_]);                                                                    \
        }                                                                                                              \
        if (AC_SCATTER)                                                                                                \
        {                                                                                                              \
            ac_fill_random(packed, total);                                                                             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            for (int32_t s_ = 0; s_ < good.num; s_++)                                                                  \
            {                                                                                                          \
                ac_fill_random(ac_slices[s_], lens[s_]);                                                               \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)
#define AC_CHECK_UNTOUCHED()                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (AC_SCATTER)                                                                                                \
        {                                                                                                              \
            for (int32_t s_ = 0; s_ < good.num; s_++)                                                                  \
            {                                                                                                          \
                ac_assert_guard(ac_slices[s_], lens[s_]);                                                              \
            }                                                                                                          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            ac_assert_guard(packed, total);                                                                            \
        }                                                                                                              \
        ac_assert_guard(ac_packed_buf, AC_GUARD);                                                                      \
        ac_assert_guard(packed + total, AC_GUARD);                                                                     \
        ac_assert_slice_guards(good.num);                                                                              \
    } while (0)

    ac_seed = 0xE0u + AC_OP;
    ac_case_t c;

    /* Sanity: the good case succeeds through the same path. */
    AC_PREP();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, ac_call(&good, packed, ac_slices));

    /* axis out of range (negative, == dims). */
    c = good;
    c.axis = -1;
    AC_PREP();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&c, packed, ac_slices));
    AC_CHECK_UNTOUCHED();
    /* axis == rank (one past the end). */
    {
#if AC_OP == AC_OP_PACK
        int32_t kshape[2] = {good.shape[0], good.shape[2]};
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          AC_KERNEL((const ac_t *const *)ac_slices, good.num, 2, kshape, 3, packed));
#else
        c = good;
        c.axis = good.dims;
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&c, packed, ac_slices));
#endif
        AC_CHECK_UNTOUCHED();
    }

    /* Rank 0 (rank-0 is only legal for pack) and negative rank. */
    {
        int32_t kshape[1] = {good.shape[0]};
        AC_PREP();
#if AC_OP == AC_OP_SPLIT
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 0, kshape, 0, good.num, good.sizes, ac_slices));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, -1, kshape, 0, good.num, good.sizes, ac_slices));
#elif AC_OP == AC_OP_UNPACK
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 0, kshape, 0, ac_slices));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, -1, kshape, 0, ac_slices));
#elif AC_OP == AC_OP_CONCAT
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          AC_KERNEL((const ac_t *const *)ac_slices, good.num, good.sizes, 0, kshape, 0, packed));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          AC_KERNEL((const ac_t *const *)ac_slices, good.num, good.sizes, -1, kshape, 0, packed));
#else
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          AC_KERNEL((const ac_t *const *)ac_slices, good.num, -1, kshape, 0, packed));
#endif
        AC_CHECK_UNTOUCHED();
    }

    /* Negative shape entry. */
    c = good;
    c.shape[2] = -5;
    AC_PREP();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&c, packed, ac_slices));
    AC_CHECK_UNTOUCHED();

#if !AC_UNIT
    /* Slice sizes that do not sum to the axis extent, a negative size, and zero slices. */
    c = good;
    c.sizes[2] += 1;
    AC_PREP();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&c, packed, ac_slices));
    AC_CHECK_UNTOUCHED();
    c = good;
    c.sizes[0] = -1;
    c.sizes[1] = 4;
    AC_PREP();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&c, packed, ac_slices));
    AC_CHECK_UNTOUCHED();
    c = good;
    c.num = 0;
    AC_PREP();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&c, packed, ac_slices));
    AC_CHECK_UNTOUCHED();
#else
    /* Zero slices along the unit axis (unpack: axis extent 0; pack: num_inputs 0). */
    {
        int32_t kshape[2] = {good.shape[0], good.shape[2]};
        AC_PREP();
    #if AC_OP == AC_OP_UNPACK
        int32_t zshape[3] = {good.shape[0], 0, good.shape[2]};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 3, zshape, 1, ac_slices));
        (void)kshape;
    #else
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL((const ac_t *const *)ac_slices, 0, 2, kshape, 1, packed));
    #endif
        AC_CHECK_UNTOUCHED();
    }
#endif

    /* Element counts above INT32_MAX: the int64 product overflow shape (every factor near INT32_MAX) and a
     * plain 2^32-2 count must both fail with nothing written; a count of 0 with factors at INT32_MAX validates. */
    {
        const int32_t big = INT32_MAX;
#if AC_OP == AC_OP_SPLIT
        int32_t s_over[3] = {big, big, 1 << 30}, s_wrap[3] = {2, big, 1}, s_edge[3] = {1, big, 0};
        int32_t one[1] = {big}, two[2] = {big - 1, 1};
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 3, s_over, 1, 1, one, ac_slices));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 3, s_wrap, 1, 1, one, ac_slices));
        AC_CHECK_UNTOUCHED();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AC_KERNEL(packed, 3, s_edge, 1, 2, two, ac_slices));
        AC_CHECK_UNTOUCHED();
#elif AC_OP == AC_OP_UNPACK
        int32_t s_over[3] = {big, 2, 1 << 30}, s_wrap[3] = {2, 3, big}, s_edge[4] = {1, 3, big, 0};
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 3, s_over, 1, ac_slices));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL(packed, 3, s_wrap, 1, ac_slices));
        AC_CHECK_UNTOUCHED();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AC_KERNEL(packed, 4, s_edge, 1, ac_slices));
        AC_CHECK_UNTOUCHED();
#elif AC_OP == AC_OP_CONCAT
        int32_t s_over[3] = {big, big, 1 << 30}, s_wrap[3] = {2, big, 1}, s_edge[3] = {1, big, 0};
        int32_t one[1] = {big}, two[2] = {big - 1, 1};
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          AC_KERNEL((const ac_t *const *)ac_slices, 1, one, 3, s_over, 1, packed));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          AC_KERNEL((const ac_t *const *)ac_slices, 1, one, 3, s_wrap, 1, packed));
        AC_CHECK_UNTOUCHED();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          AC_KERNEL((const ac_t *const *)ac_slices, 2, two, 3, s_edge, 1, packed));
        AC_CHECK_UNTOUCHED();
#else
        int32_t s_over[2] = {big, 1 << 30}, s_wrap[1] = {1 << 30}, s_edge[2] = {big, 0};
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL((const ac_t *const *)ac_slices, 2, 2, s_over, 0, packed));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AC_KERNEL((const ac_t *const *)ac_slices, 4, 1, s_wrap, 0, packed));
        AC_CHECK_UNTOUCHED();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AC_KERNEL((const ac_t *const *)ac_slices, 2, 2, s_edge, 0, packed));
        AC_CHECK_UNTOUCHED();
#endif
    }

    /* NULL pointers. */
    {
        ac_t *null_slices[AC_MAX_SLICES];
        for (int32_t s = 0; s < good.num; s++)
        {
            null_slices[s] = ac_slices[s];
        }
        null_slices[1] = NULL;
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&good, packed, null_slices));
        AC_CHECK_UNTOUCHED();
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&good, NULL, ac_slices));
        AC_CHECK_UNTOUCHED();
        AC_PREP();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, ac_call(&good, packed, NULL));
        AC_CHECK_UNTOUCHED();
    }
#undef AC_PREP
#undef AC_CHECK_UNTOUCHED
}

/* Random ranks, shapes, axes and slices. */
void AC_FN(fuzz)(void)
{
    ac_seed = 0xF00Du + AC_OP;
    for (int32_t it = 0; it < AC_FUZZ_ITERS; it++)
    {
        ac_case_t c;
        const int32_t dims = ac_rand_range(1, 5);
        ac_random_case(&c, dims, ac_rand_range(0, dims - 1), 1024);
        ac_run_case(&c);
    }
}

#if AC_OP == AC_OP_CONCAT
/* At rank 4 the result must equal what the per-axis 4-D kernels produce for the same inputs. */
void AC_FN(matches_4d_axis_kernels)(void)
{
    static ac_t ref[AC_MAX_ELEMENTS];
    ac_seed = 0x4D4Du;
    for (int32_t axis = 0; axis < 4; axis++)
    {
        for (int32_t rep = 0; rep < 4; rep++)
        {
            ac_case_t c;
            ac_random_case(&c, 4, axis, AC_MAX_ELEMENTS);
            ac_run_case(&c); /* also lays out random slices in the arena */
            const int32_t total = ac_product(c.shape, c.dims);
            ac_t *packed = ac_packed_buf + AC_GUARD;
            /* Row-major [w, z, y, x]: axis 3 is x. */
            const int32_t out_w = c.shape[0], out_z = c.shape[1], out_y = c.shape[2], out_x = c.shape[3];
            uint32_t offset = 0;
            for (int32_t s = 0; s < c.num; s++)
            {
                const int32_t iw = (axis == 0) ? c.sizes[s] : out_w;
                const int32_t iz = (axis == 1) ? c.sizes[s] : out_z;
                const int32_t iy = (axis == 2) ? c.sizes[s] : out_y;
                const int32_t ix = (axis == 3) ? c.sizes[s] : out_x;
                switch (axis)
                {
                case 3:
                    AC_CAT(AC_KERNEL, x)(ac_slices[s], ix, iy, iz, iw, ref, out_x, offset);
                    break;
                case 2:
                    AC_CAT(AC_KERNEL, y)(ac_slices[s], ix, iy, iz, iw, ref, out_y, offset);
                    break;
                case 1:
                    AC_CAT(AC_KERNEL, z)(ac_slices[s], ix, iy, iz, iw, ref, out_z, offset);
                    break;
                default:
                    AC_CAT(AC_KERNEL, w)(ac_slices[s], ix, iy, iz, iw, ref, offset);
                    break;
                }
                offset += (uint32_t)c.sizes[s];
            }
            for (int32_t i = 0; i < total; i++)
            {
                AC_ASSERT_BITS(ac_bits_of(ref[i]), ac_bits_of(packed[i]));
            }
        }
    }
}
#endif
