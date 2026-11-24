/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* The rbf kernel stage of Internal/arm_nn_rbf_kernel_flt.h and its entry arm_nn_rbf_kernel_f32, against the
 * references of rbf_kernel_reference.h, row by row.
 *
 *   rk_sum_normal      every lane of arm_nn_rbf_sq_distance_block_f32, called a block of four rows at a time, against
 *                      the reference sum of the form compiled (ARM_NN_RBF_KERNEL_FUSED: fmaf a term; otherwise the
 *                      square rounded, then the add) on normal values, over every remainder of the rows and the
 *                      features modulo four and randomised shapes. The other form's sum is counted beside it, and
 *                      the distance between the two forms' references (lanes that differ, the largest difference in
 *                      units in the last place) is reported. A digest of every lane's bits is reported, so two builds
 *                      of the same seeds, one per leg, compare bit for bit.
 *   rk_stage_normal    the kernel values of the block body, the stage body and the entry on the same cases, each
 *                      lane against exp(reference sum * neg_gamma) with the exponential of the build's leg, held; on
 *                      the Helium leg also against the scalar exponential, counted with the largest difference in
 *                      units in the last place (the exponential's own design: Helium flushes a subnormal result);
 *                      and against the other form's reference, counted. neg_gamma is set per case from the first
 *                      row's sum so that the arguments spread over [-120, 0]: normal, subnormal and +0 results all
 *                      occur, and their counts are reported. Two digests: of every lane's bits, and of every lane
 *                      with a subnormal result read as +0, which two legs share.
 *   rk_literal         the stage body called with literal rows and features, as a generated module calls it.
 *   rk_entry_refusals  every argument the entry refuses (NULL arrays, rows or features below one, a table above
 *                      INT32_MAX bytes, an output overlapping the input or the table) refused with nothing written.
 *   rk_special         a NaN of four encodings, both infinities, both zeros, both subnormals planted in the input or
 *                      a row; neg_gamma of either zero, NaN, either infinity, a subnormal, a large positive value; a
 *                      subnormal difference, a subnormal square, a square that overflows and a distance that
 *                      underflows the exponential: the lanes whose sum or kernel value differs from the reference,
 *                      counted by kind and site and reported, not held. Every lane the planted value does not reach
 *                      is held as on normal values.
 *
 * The normal values have exponents in [-6, 6], so no difference, square or partial sum is subnormal. Every write of a
 * body past its last lane is seen by guard words; the table and the input sit at the end of arrays of their largest
 * size, so a read past one is seen by AddressSanitizer on the host. Under fast-math (RK_FAST_MATH) the compiler may
 * reorder a body compiled in this unit: the lane equalities are then counted and reported, and the guards and the
 * refusals are still held. The multiply-then-add form has a fixed rounding only where the build does not contract;
 * a build says so by defining RBF_TEST_NO_CONTRACT (with -ffp-contract=off, or Arm GNU in an ISO dialect), and only
 * there is that form held. */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "../../../../Include/Internal/arm_nn_rbf_kernel_flt.h"
#include "rbf_kernel_reference.h"

#define RK_MAX_ROWS 72
#define RK_MAX_FEATURES 40
#define RK_TABLE_WORDS (RK_MAX_ROWS * RK_MAX_FEATURES)
#define RK_GUARD 4
#define RK_SENTINEL UINT32_C(0x5a5a5a5a)

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define RK_HELIUM 1
    #define RK_LEG "helium"
#else
    #define RK_HELIUM 0
    #define RK_LEG "scalar"
#endif

/* Reassociation allowed: GCC says so by __ASSOCIATIVE_MATH__ and Clang by __ARM_FP_FAST, each also under
 * -fno-finite-math-only, which takes __FAST_MATH__ away. */
#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__) || defined(__ARM_FP_FAST)
    #define RK_FAST_MATH 1
#else
    #define RK_FAST_MATH 0
#endif

#if defined(RBF_TEST_NO_CONTRACT)
    #define RK_NO_CONTRACT 1
#else
    #define RK_NO_CONTRACT 0
#endif

/* The form compiled is held where its rounding is fixed: always for the fused form, only in a build that does not
 * contract for the multiply-then-add form; never under fast-math. */
#define RK_HOLD_FORM (!RK_FAST_MATH && (ARM_NN_RBF_KERNEL_FUSED || RK_NO_CONTRACT))
#define RK_FORM (ARM_NN_RBF_KERNEL_FUSED ? "fused" : "mul-add")
#define RK_OTHER_FORM (ARM_NN_RBF_KERNEL_FUSED ? "mul-add" : "fused")

static char rk_message[400];

/* ---- Words and values ---- */

static uint32_t rk_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float32_t rk_float(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int rk_is_nan(uint32_t bits) { return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000); }

static int rk_is_subnormal(uint32_t bits)
{
    return (bits & UINT32_C(0x7f800000)) == 0 && (bits & UINT32_C(0x007fffff)) != 0;
}

static int rk_is_normal(uint32_t bits)
{
    const uint32_t exponent = bits & UINT32_C(0x7f800000);
    return exponent != 0 && exponent != UINT32_C(0x7f800000);
}

/* The distance in units in the last place between two non-NaN words, on the line of every float32 in order (-0 and
 * +0 one apart); a NaN on either side counts as the largest distance. */
static uint32_t rk_ulps(uint32_t a, uint32_t b)
{
    if (rk_is_nan(a) || rk_is_nan(b))
        return rk_is_nan(a) && rk_is_nan(b) ? 0u : UINT32_MAX;
    const int64_t ka = (a & UINT32_C(0x80000000)) ? -(int64_t)(a & UINT32_C(0x7fffffff)) - 1 : (int64_t)a;
    const int64_t kb = (b & UINT32_C(0x80000000)) ? -(int64_t)(b & UINT32_C(0x7fffffff)) - 1 : (int64_t)b;
    const int64_t d = ka > kb ? ka - kb : kb - ka;
    return d > (int64_t)UINT32_MAX ? UINT32_MAX : (uint32_t)d;
}

static uint32_t rk_state;

static void rk_seed(uint32_t seed) { rk_state = seed ? seed : UINT32_C(0x9e3779b9); }

static uint32_t rk_next(void)
{
    uint32_t x = rk_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rk_state = x;
    return x;
}

static int32_t rk_below(int32_t bound) { return (int32_t)(rk_next() % (uint32_t)bound); }

/* A normal value of either sign with an exponent in [-6, 6] and a random significand. */
static float32_t rk_normal(void)
{
    const uint32_t r = rk_next();
    const uint32_t exponent = (uint32_t)(127 - 6 + (int32_t)((r >> 24) % 13u));
    return rk_float((r & UINT32_C(0x80000000)) | (exponent << 23) | (rk_next() & UINT32_C(0x007fffff)));
}

/* ---- The buffers: the table and the input each in use at the end of an array of its largest size ---- */

static float32_t rk_table_store[RK_TABLE_WORDS];
static float32_t rk_input_store[RK_MAX_FEATURES];
static float32_t rk_out[RK_MAX_ROWS + RK_GUARD];

typedef struct
{
    int32_t rows;
    int32_t features;
    float32_t neg_gamma;
    float32_t *table; /* rows * features, row-major */
    float32_t *input;
} rk_case;

static rk_case rk_place(int32_t rows, int32_t features)
{
    rk_case c;
    c.rows = rows;
    c.features = features;
    c.neg_gamma = -1.0f;
    c.table = rk_table_store + (RK_TABLE_WORDS - rows * features);
    c.input = rk_input_store + (RK_MAX_FEATURES - features);
    return c;
}

/* Normal values, then neg_gamma such that the first row's argument is near -t, t in [0, 120): the other rows'
 * arguments spread about it, over the exponential's normal, subnormal and zero results. */
static void rk_fill_normal(rk_case *c)
{
    for (int32_t i = 0; i < c->rows * c->features; ++i)
        c->table[i] = rk_normal();
    for (int32_t k = 0; k < c->features; ++k)
        c->input[k] = rk_normal();
    const float32_t sum = rk_reference_sum_fused(c->table, c->input, c->features);
    const float32_t target = (float32_t)rk_below(1200) * 0.1f;
    c->neg_gamma = sum > 0.0f ? -target / sum : -1.0f;
}

static void rk_clear_out(void)
{
    for (int32_t i = 0; i < RK_MAX_ROWS + RK_GUARD; ++i)
        rk_out[i] = rk_float(RK_SENTINEL);
}

/* 1 if nothing at or past rk_out[lanes] was written. */
static int rk_guards_hold(int32_t lanes)
{
    for (int32_t i = lanes; i < RK_MAX_ROWS + RK_GUARD; ++i)
        if (rk_bits(rk_out[i]) != RK_SENTINEL)
            return 0;
    return 1;
}

/* ---- The references of one row ---- */

typedef struct
{
    uint32_t sum[2];        /* [0] the form compiled, [1] the other form */
    uint32_t kernel[2];     /* exp of each with the leg's exponential */
    uint32_t kernel_scalar; /* the form compiled, the scalar exponential */
} rk_reference;

static void rk_reference_of(const rk_case *c, int32_t row, rk_reference *r)
{
    const float32_t *const values = c->table + row * c->features;
    const float32_t fused = rk_reference_sum_fused(values, c->input, c->features);
    const float32_t unfused = rk_reference_sum_unfused(values, c->input, c->features);
    const float32_t form = ARM_NN_RBF_KERNEL_FUSED ? fused : unfused;
    const float32_t other = ARM_NN_RBF_KERNEL_FUSED ? unfused : fused;
    r->sum[0] = rk_bits(form);
    r->sum[1] = rk_bits(other);
    r->kernel[0] = rk_bits(rk_reference_exp_leg(rk_reference_argument(form, c->neg_gamma)));
    r->kernel[1] = rk_bits(rk_reference_exp_leg(rk_reference_argument(other, c->neg_gamma)));
    r->kernel_scalar = rk_bits(rk_reference_exp_scalar(rk_reference_argument(form, c->neg_gamma)));
}

/* ---- Lane tallies ---- */

typedef struct
{
    uint32_t lanes;
    uint32_t equal;                 /* bits equal to the held reference */
    uint32_t nan_differs;           /* NaN-ness differs from the held reference */
    uint32_t other_differs;         /* bits differ from the second reference */
    uint32_t other_max_ulps;        /* the largest distance from it */
    uint32_t other_max_ulps_normal; /* the same over lanes where both are normal numbers */
    uint32_t guard_failures;
    uint32_t first_got, first_want;                       /* the first lane that differs from the held reference */
    uint32_t digest;                                      /* FNV-1a over every lane's bits in order */
    uint32_t flushed_digest;                              /* the same with a subnormal lane read as +0 */
    uint32_t zero, subnormal, one, normal, infinite, nan; /* the held reference's results */
} rk_tally;

static uint32_t rk_fnv(uint32_t digest, uint32_t word)
{
    for (int byte = 0; byte < 4; ++byte)
        digest = (digest ^ ((word >> (8 * byte)) & 0xffu)) * UINT32_C(16777619);
    return digest;
}

static void rk_count(rk_tally *t, uint32_t got, uint32_t want, uint32_t other)
{
    if (t->lanes == 0)
    {
        t->digest = UINT32_C(2166136261);
        t->flushed_digest = UINT32_C(2166136261);
    }
    if (t->lanes == t->equal && got != want)
    {
        t->first_got = got;
        t->first_want = want;
    }
    t->digest = rk_fnv(t->digest, got);
    t->flushed_digest = rk_fnv(t->flushed_digest, rk_is_subnormal(got) ? 0u : got);
    ++t->lanes;
    t->equal += got == want;
    t->nan_differs += rk_is_nan(got) != rk_is_nan(want);
    t->other_differs += got != other;
    const uint32_t u = rk_ulps(got, other);
    t->other_max_ulps = u > t->other_max_ulps ? u : t->other_max_ulps;
    if (rk_is_normal(got) && rk_is_normal(other))
        t->other_max_ulps_normal = u > t->other_max_ulps_normal ? u : t->other_max_ulps_normal;
    t->zero += (want & UINT32_C(0x7fffffff)) == 0;
    t->subnormal += rk_is_subnormal(want);
    t->one += want == UINT32_C(0x3f800000);
    t->infinite += (want & UINT32_C(0x7fffffff)) == UINT32_C(0x7f800000);
    t->nan += rk_is_nan(want);
    t->normal += (want & UINT32_C(0x7f800000)) != 0 && (want & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) &&
        want != UINT32_C(0x3f800000);
}

/* The first `lanes` values of a block from `first`. */
static int32_t rk_lanes_from(const rk_case *c, int32_t first)
{
    return c->rows - first < ARM_NN_RBF_BLOCK_ROWS ? c->rows - first : ARM_NN_RBF_BLOCK_ROWS;
}

/* Every block of one case through the squared-distance body. */
static void rk_run_sum(const rk_case *c, rk_tally *t, rk_tally *forms)
{
    for (int32_t first = 0; first < c->rows; first += ARM_NN_RBF_BLOCK_ROWS)
    {
        const int32_t lanes = rk_lanes_from(c, first);
        rk_clear_out();
        arm_nn_rbf_sq_distance_block_f32(c->table + first * c->features, c->input, c->features, lanes, rk_out);
        t->guard_failures += !rk_guards_hold(lanes);
        for (int32_t j = 0; j < lanes; ++j)
        {
            rk_reference r;
            rk_reference_of(c, first + j, &r);
            rk_count(t, rk_bits(rk_out[j]), r.sum[0], r.sum[1]);
            rk_count(forms, r.sum[1], r.sum[0], r.sum[0]);
        }
    }
}

enum
{
    RK_BLOCK,
    RK_BODY,
    RK_ENTRY,
    RK_PATHS
};

static const char *const rk_path_name[RK_PATHS] = {"block", "body", "entry"};

/* One case through one path, every lane tallied against the leg's reference [0] and the scalar exponential [1]; the
 * forms tally takes the other form's reference against the form's. */
static void rk_run_stage(int path, const rk_case *c, rk_tally *t, rk_tally *scalar, rk_tally *forms)
{
    rk_clear_out();
    if (path == RK_BLOCK)
    {
        for (int32_t first = 0; first < c->rows; first += ARM_NN_RBF_BLOCK_ROWS)
        {
            const int32_t lanes = rk_lanes_from(c, first);
            arm_nn_rbf_kernel_block_f32(
                c->table + first * c->features, c->input, c->neg_gamma, c->features, lanes, rk_out + first);
            /* The guards of a block are the next rows' places, not yet written, and those after the table. */
            t->guard_failures += !rk_guards_hold(first + lanes);
        }
    }
    else if (path == RK_BODY)
    {
        arm_nn_rbf_kernel_body_f32(c->input, c->table, c->rows, c->features, c->neg_gamma, rk_out);
        t->guard_failures += !rk_guards_hold(c->rows);
    }
    else
    {
        t->guard_failures += arm_nn_rbf_kernel_f32(c->input, c->table, c->rows, c->features, c->neg_gamma, rk_out) !=
            ARM_CMSIS_NN_SUCCESS;
        t->guard_failures += !rk_guards_hold(c->rows);
    }
    for (int32_t s = 0; s < c->rows; ++s)
    {
        rk_reference r;
        rk_reference_of(c, s, &r);
        const uint32_t got = rk_bits(rk_out[s]);
        rk_count(t, got, r.kernel[0], r.kernel[1]);
        rk_count(scalar, got, r.kernel_scalar, r.kernel_scalar);
        if (path == RK_BODY)
            rk_count(forms, r.kernel[1], r.kernel[0], r.kernel[0]);
    }
}

/* Every remainder of the rows and the features modulo four, then randomised shapes, through `visit`. */
static uint32_t rk_shapes(uint32_t seed, void (*visit)(const rk_case *, void *), void *state)
{
    uint32_t cases = 0;
    rk_seed(seed);
    for (int32_t features = 1; features <= 24; ++features)
        for (int32_t rows = 1; rows <= 13; ++rows)
            for (int trial = 0; trial < 2; ++trial)
            {
                rk_case c = rk_place(rows, features);
                rk_fill_normal(&c);
                visit(&c, state);
                ++cases;
            }
    for (int trial = 0; trial < 300; ++trial)
    {
        rk_case c = rk_place(1 + rk_below(RK_MAX_ROWS), 1 + rk_below(RK_MAX_FEATURES));
        rk_fill_normal(&c);
        visit(&c, state);
        ++cases;
    }
    return cases;
}

/* ---- The cases ---- */

static void rk_capability(void)
{
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_RBF_KERNEL_F32);
    TEST_ASSERT_EQUAL_INT(4, ARM_NN_RBF_BLOCK_ROWS);
    snprintf(rk_message,
             sizeof(rk_message),
             "rbf build: %s leg, %s form%s%s",
             RK_LEG,
             RK_FORM,
             RK_FAST_MATH ? ", fast-math" : "",
             RK_NO_CONTRACT ? ", no contraction" : "");
    TEST_MESSAGE(rk_message);
}

typedef struct
{
    rk_tally sum;
    rk_tally forms;
} rk_sum_state;

static void rk_visit_sum(const rk_case *c, void *state)
{
    rk_sum_state *const s = (rk_sum_state *)state;
    rk_run_sum(c, &s->sum, &s->forms);
}

static void rk_sum_normal(void)
{
    rk_sum_state s;
    memset(&s, 0, sizeof(s));
    const uint32_t cases = rk_shapes(UINT32_C(0x5eed0f1a), rk_visit_sum, &s);
    snprintf(rk_message,
             sizeof(rk_message),
             "rbf sum %s %s%s: %lu cases, %lu lanes, equal %s %lu, differ from %s %lu (largest %lu ulp), digest "
             "0x%08lx, guard failures %lu, first difference 0x%08lx against 0x%08lx",
             RK_LEG,
             RK_FORM,
             RK_FAST_MATH ? " fast-math" : "",
             (unsigned long)cases,
             (unsigned long)s.sum.lanes,
             RK_FORM,
             (unsigned long)s.sum.equal,
             RK_OTHER_FORM,
             (unsigned long)s.sum.other_differs,
             (unsigned long)s.sum.other_max_ulps,
             (unsigned long)s.sum.digest,
             (unsigned long)s.sum.guard_failures,
             (unsigned long)s.sum.first_got,
             (unsigned long)s.sum.first_want);
    TEST_MESSAGE(rk_message);
    snprintf(rk_message,
             sizeof(rk_message),
             "rbf forms sum: references of the two forms differ on %lu of %lu lanes, largest %lu ulp",
             (unsigned long)(s.forms.lanes - s.forms.equal),
             (unsigned long)s.forms.lanes,
             (unsigned long)s.forms.other_max_ulps);
    TEST_MESSAGE(rk_message);
    TEST_ASSERT_EQUAL_UINT32(0, s.sum.guard_failures);
#if RK_HOLD_FORM
    TEST_ASSERT_EQUAL_UINT32(s.sum.lanes, s.sum.equal);
#endif
}

typedef struct
{
    rk_tally leg[RK_PATHS];
    rk_tally scalar[RK_PATHS];
    rk_tally forms;
} rk_stage_state;

static void rk_visit_stage(const rk_case *c, void *state)
{
    rk_stage_state *const s = (rk_stage_state *)state;
    for (int path = 0; path < RK_PATHS; ++path)
        rk_run_stage(path, c, &s->leg[path], &s->scalar[path], &s->forms);
}

static void rk_stage_normal(void)
{
    rk_stage_state s;
    memset(&s, 0, sizeof(s));
    const uint32_t cases = rk_shapes(UINT32_C(0x5eed0f1a), rk_visit_stage, &s);
    for (int path = 0; path < RK_PATHS; ++path)
    {
        const rk_tally *const t = &s.leg[path];
        const rk_tally *const u = &s.scalar[path];
        snprintf(rk_message,
                 sizeof(rk_message),
                 "rbf stage %s %s %s%s: %lu cases, %lu lanes (%lu +0, %lu subnormal, %lu one, %lu other normal), "
                 "equal %lu, differ from the scalar exponential %lu (largest %lu ulp), digest 0x%08lx, flushed "
                 "digest 0x%08lx, guard failures %lu, first difference 0x%08lx against 0x%08lx",
                 rk_path_name[path],
                 RK_LEG,
                 RK_FORM,
                 RK_FAST_MATH ? " fast-math" : "",
                 (unsigned long)cases,
                 (unsigned long)t->lanes,
                 (unsigned long)t->zero,
                 (unsigned long)t->subnormal,
                 (unsigned long)t->one,
                 (unsigned long)t->normal,
                 (unsigned long)t->equal,
                 (unsigned long)(u->lanes - u->equal),
                 (unsigned long)u->other_max_ulps,
                 (unsigned long)t->digest,
                 (unsigned long)t->flushed_digest,
                 (unsigned long)t->guard_failures,
                 (unsigned long)t->first_got,
                 (unsigned long)t->first_want);
        TEST_MESSAGE(rk_message);
        snprintf(rk_message,
                 sizeof(rk_message),
                 "rbf stage %s %s: differ from the %s reference %lu (largest %lu ulp, %lu between normal results)",
                 rk_path_name[path],
                 RK_LEG,
                 RK_OTHER_FORM,
                 (unsigned long)t->other_differs,
                 (unsigned long)t->other_max_ulps,
                 (unsigned long)t->other_max_ulps_normal);
        TEST_MESSAGE(rk_message);
        TEST_ASSERT_EQUAL_UINT32_MESSAGE(0, t->guard_failures, rk_path_name[path]);
#if RK_HOLD_FORM
        TEST_ASSERT_EQUAL_UINT32_MESSAGE(t->lanes, t->equal, rk_path_name[path]);
#endif
    }
    snprintf(rk_message,
             sizeof(rk_message),
             "rbf forms stage %s: kernel values of the two forms' references differ on %lu of %lu lanes, largest %lu "
             "ulp (%lu between normal results)",
             RK_LEG,
             (unsigned long)(s.forms.lanes - s.forms.equal),
             (unsigned long)s.forms.lanes,
             (unsigned long)s.forms.other_max_ulps,
             (unsigned long)s.forms.other_max_ulps_normal);
    TEST_MESSAGE(rk_message);
    /* Every path computes the same values where the form's rounding is fixed; elsewhere the entry's translation unit
     * may round differently from this one's, and the digests above are reported. */
#if RK_HOLD_FORM
    TEST_ASSERT_EQUAL_HEX32(s.leg[RK_BLOCK].digest, s.leg[RK_BODY].digest);
    TEST_ASSERT_EQUAL_HEX32(s.leg[RK_BLOCK].digest, s.leg[RK_ENTRY].digest);
#endif
    /* The cases reach the exponential's normal, subnormal and zero results (by the scalar exponential, whose
     * subnormal results the Helium one returns as +0). */
    TEST_ASSERT_TRUE(s.scalar[RK_BODY].zero > 0 && s.scalar[RK_BODY].subnormal > 0 && s.scalar[RK_BODY].normal > 0);
}

/* The stage body with literal extents, as a generated module calls it, each lane held to the reference. */
static uint32_t rk_literal_case(int32_t rows, int32_t features, uint32_t *lanes)
{
    rk_case c = rk_place(rows, features);
    rk_fill_normal(&c);
    rk_clear_out();
    if (rows == 7 && features == 5)
        arm_nn_rbf_kernel_body_f32(c.input, c.table, 7, 5, c.neg_gamma, rk_out);
    else if (rows == 3 && features == 16)
        arm_nn_rbf_kernel_body_f32(c.input, c.table, 3, 16, c.neg_gamma, rk_out);
    else
        arm_nn_rbf_kernel_body_f32(c.input, c.table, 8, 1, c.neg_gamma, rk_out);
    uint32_t failures = (uint32_t)!rk_guards_hold(rows);
    for (int32_t s = 0; s < rows; ++s)
    {
        rk_reference r;
        rk_reference_of(&c, s, &r);
        failures += RK_HOLD_FORM && rk_bits(rk_out[s]) != r.kernel[0];
        ++*lanes;
    }
    return failures;
}

static void rk_literal(void)
{
    uint32_t failures = 0, lanes = 0;
    rk_seed(UINT32_C(0x11e7a1));
    for (int trial = 0; trial < 40; ++trial)
    {
        failures += rk_literal_case(7, 5, &lanes);
        failures += rk_literal_case(3, 16, &lanes);
        failures += rk_literal_case(8, 1, &lanes);
    }
    snprintf(rk_message,
             sizeof(rk_message),
             "rbf literal %s %s: %lu lanes, failures %lu",
             RK_LEG,
             RK_FORM,
             (unsigned long)lanes,
             (unsigned long)failures);
    TEST_MESSAGE(rk_message);
    TEST_ASSERT_EQUAL_UINT32(0, failures);
}

static void rk_entry_refusals(void)
{
    rk_case c = rk_place(6, 3);
    rk_fill_normal(&c);
    const float32_t g = c.neg_gamma;
    float32_t *const out = rk_out;
    struct
    {
        const float32_t *input;
        const float32_t *table;
        int32_t rows;
        int32_t features;
        float32_t *output;
    } refused[] = {
        {NULL, c.table, 6, 3, out},
        {c.input, NULL, 6, 3, out},
        {c.input, c.table, 6, 3, NULL},
        {c.input, c.table, 0, 3, out},
        {c.input, c.table, -1, 3, out},
        {c.input, c.table, 6, 0, out},
        {c.input, c.table, 6, -2, out},
        {c.input, c.table, INT32_MAX / 4 + 1, 1, out},
        {c.input, c.table, 1 << 15, 1 << 14, out},
        {c.input, c.table, 6, 3, c.input + 1},
        {c.input, c.table, 6, 3, c.table + 5},
        {c.input, c.table, 6, 3, c.table - 3},
    };
    static float32_t table_copy[RK_TABLE_WORDS];
    static float32_t input_copy[RK_MAX_FEATURES];
    memcpy(table_copy, rk_table_store, sizeof(table_copy));
    memcpy(input_copy, rk_input_store, sizeof(input_copy));
    uint32_t written = 0;
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i)
    {
        rk_clear_out();
        TEST_ASSERT_EQUAL_INT(
            ARM_CMSIS_NN_ARG_ERROR,
            arm_nn_rbf_kernel_f32(
                refused[i].input, refused[i].table, refused[i].rows, refused[i].features, g, refused[i].output));
        written += !rk_guards_hold(0);
        written += memcmp(table_copy, rk_table_store, sizeof(table_copy)) != 0 ||
            memcmp(input_copy, rk_input_store, sizeof(input_copy)) != 0;
    }
    TEST_ASSERT_EQUAL_UINT32(0, written);
    rk_clear_out();
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_rbf_kernel_f32(c.input, c.table, 6, 3, g, out));
    TEST_ASSERT_TRUE(rk_guards_hold(6));
}

/* ---- Special values ---- */

enum
{
    RK_SITE_INPUT,
    RK_SITE_ROW,
    RK_SITE_GAMMA,
    RK_SITE_PAIR /* a row's feature and the input's together */
};

static const char *const rk_site_name[] = {"input", "row", "neg_gamma", "row and input"};

typedef struct
{
    const char *name;
    int site;
    uint32_t bits;  /* the value planted; for RK_SITE_PAIR the row's */
    uint32_t other; /* for RK_SITE_PAIR the input's */
    int zero_gamma; /* neg_gamma set to +0 as well */
} rk_kind;

static const rk_kind rk_kinds[] = {
    {"+qNaN", RK_SITE_INPUT, UINT32_C(0x7fc00000), 0, 0},
    {"-qNaN", RK_SITE_INPUT, UINT32_C(0xffc00000), 0, 0},
    {"qNaN payload", RK_SITE_INPUT, UINT32_C(0x7fc12345), 0, 0},
    {"sNaN", RK_SITE_INPUT, UINT32_C(0x7f812345), 0, 0},
    {"+inf", RK_SITE_INPUT, UINT32_C(0x7f800000), 0, 0},
    {"-inf", RK_SITE_INPUT, UINT32_C(0xff800000), 0, 0},
    {"+0", RK_SITE_INPUT, UINT32_C(0x00000000), 0, 0},
    {"-0", RK_SITE_INPUT, UINT32_C(0x80000000), 0, 0},
    {"+subnormal", RK_SITE_INPUT, UINT32_C(0x00012345), 0, 0},
    {"-subnormal", RK_SITE_INPUT, UINT32_C(0x80012345), 0, 0},
    {"+qNaN", RK_SITE_ROW, UINT32_C(0x7fc00000), 0, 0},
    {"-qNaN", RK_SITE_ROW, UINT32_C(0xffc00000), 0, 0},
    {"qNaN payload", RK_SITE_ROW, UINT32_C(0x7fc12345), 0, 0},
    {"sNaN", RK_SITE_ROW, UINT32_C(0x7f812345), 0, 0},
    {"+inf", RK_SITE_ROW, UINT32_C(0x7f800000), 0, 0},
    {"-inf", RK_SITE_ROW, UINT32_C(0xff800000), 0, 0},
    {"+0", RK_SITE_ROW, UINT32_C(0x00000000), 0, 0},
    {"-0", RK_SITE_ROW, UINT32_C(0x80000000), 0, 0},
    {"+subnormal", RK_SITE_ROW, UINT32_C(0x00012345), 0, 0},
    {"-subnormal", RK_SITE_ROW, UINT32_C(0x80012345), 0, 0},
    {"+0", RK_SITE_GAMMA, UINT32_C(0x00000000), 0, 0},
    {"-0", RK_SITE_GAMMA, UINT32_C(0x80000000), 0, 0},
    {"qNaN", RK_SITE_GAMMA, UINT32_C(0x7fc12345), 0, 0},
    {"+inf", RK_SITE_GAMMA, UINT32_C(0x7f800000), 0, 0},
    {"-inf", RK_SITE_GAMMA, UINT32_C(0xff800000), 0, 0},
    {"-subnormal", RK_SITE_GAMMA, UINT32_C(0x80012345), 0, 0},
    {"+1e3 (overflow)", RK_SITE_GAMMA, UINT32_C(0x447a0000), 0, 0},
    /* 2^-126 * 1.125 - 2^-126: a subnormal difference of two normal values. */
    {"subnormal difference", RK_SITE_PAIR, UINT32_C(0x00900000), UINT32_C(0x00800000), 0},
    /* 3 * 2^-70 - 2^-71 = 5 * 2^-71, whose square, 25 * 2^-142, is subnormal. */
    {"subnormal square", RK_SITE_PAIR, UINT32_C(0x1d400000), UINT32_C(0x1c000000), 0},
    /* 2^100 - 0: a square that overflows to +inf. */
    {"overflowing square", RK_SITE_PAIR, UINT32_C(0x71800000), UINT32_C(0x00000000), 0},
    /* 2^12 - 0: a sum of 2^24 or more, so that the argument underflows the exponential to +0. */
    {"underflowing distance", RK_SITE_PAIR, UINT32_C(0x45800000), UINT32_C(0x00000000), 0},
    /* The overflowing square with neg_gamma +0: +0 times +inf, a NaN argument. */
    {"overflowing square, neg_gamma +0", RK_SITE_PAIR, UINT32_C(0x71800000), UINT32_C(0x00000000), 1},
};

#define RK_KINDS ((int)(sizeof(rk_kinds) / sizeof(rk_kinds[0])))

/* Plants one kind and returns the row it reaches (-1: every row). */
static int32_t rk_plant(rk_case *c, const rk_kind *kind)
{
    const int32_t row = rk_below(c->rows);
    const int32_t feature = rk_below(c->features);
    const float32_t value = rk_float(kind->bits);
    if (kind->site == RK_SITE_INPUT)
    {
        c->input[feature] = value;
        return -1;
    }
    if (kind->site == RK_SITE_GAMMA)
    {
        c->neg_gamma = value;
        return -1;
    }
    c->table[row * c->features + feature] = value;
    if (kind->site == RK_SITE_PAIR)
        c->input[feature] = rk_float(kind->other);
    if (kind->zero_gamma)
        c->neg_gamma = 0.0f;
    return row;
}

static void rk_special_values(void)
{
    uint32_t held_failures = 0;
    uint32_t guard_failures = 0;
    rk_seed(UINT32_C(0x0badcafe));
    for (int kind = 0; kind < RK_KINDS; ++kind)
    {
        rk_tally sum, stage;
        memset(&sum, 0, sizeof(sum));
        memset(&stage, 0, sizeof(stage));
        for (int trial = 0; trial < 60; ++trial)
        {
            rk_case c = rk_place(1 + rk_below(11), 1 + rk_below(9));
            rk_fill_normal(&c);
            /* A pair plant also sets the input's feature, a normal value or +0, which the other rows read in
             * normal arithmetic: they are held. A neg_gamma of +0 reaches every row, and only the planted one is
             * counted. */
            const int32_t at = rk_plant(&c, &rk_kinds[kind]);
            rk_clear_out();
            arm_nn_rbf_kernel_body_f32(c.input, c.table, c.rows, c.features, c.neg_gamma, rk_out);
            guard_failures += !rk_guards_hold(c.rows);
            float32_t sums[RK_MAX_ROWS];
            for (int32_t first = 0; first < c.rows; first += ARM_NN_RBF_BLOCK_ROWS)
                arm_nn_rbf_sq_distance_block_f32(
                    c.table + first * c.features, c.input, c.features, rk_lanes_from(&c, first), sums + first);
            for (int32_t s = 0; s < c.rows; ++s)
            {
                rk_reference r;
                rk_reference_of(&c, s, &r);
                const uint32_t got = rk_bits(rk_out[s]);
                if (at < 0 || at == s)
                {
                    rk_count(&sum, rk_bits(sums[s]), r.sum[0], r.sum[1]);
                    rk_count(&stage, got, r.kernel[0], r.kernel_scalar);
                }
                else if (RK_HOLD_FORM && (got != r.kernel[0] || rk_bits(sums[s]) != r.sum[0]))
                {
                    ++held_failures;
                }
            }
        }
        snprintf(rk_message,
                 sizeof(rk_message),
                 "rbf special %s %s %s: %lu lanes reached, sum bits differ %lu (NaN-ness %lu, first 0x%08lx against "
                 "0x%08lx), kernel bits differ %lu (NaN-ness %lu, first 0x%08lx against 0x%08lx), differ from the "
                 "scalar exponential %lu, results %lu +0 %lu subnormal %lu one %lu inf %lu NaN %lu other normal",
                 RK_LEG,
                 rk_kinds[kind].name,
                 rk_site_name[rk_kinds[kind].site],
                 (unsigned long)stage.lanes,
                 (unsigned long)(sum.lanes - sum.equal),
                 (unsigned long)sum.nan_differs,
                 (unsigned long)sum.first_got,
                 (unsigned long)sum.first_want,
                 (unsigned long)(stage.lanes - stage.equal),
                 (unsigned long)stage.nan_differs,
                 (unsigned long)stage.first_got,
                 (unsigned long)stage.first_want,
                 (unsigned long)stage.other_differs,
                 (unsigned long)stage.zero,
                 (unsigned long)stage.subnormal,
                 (unsigned long)stage.one,
                 (unsigned long)stage.infinite,
                 (unsigned long)stage.nan,
                 (unsigned long)stage.normal);
        TEST_MESSAGE(rk_message);
    }
    TEST_ASSERT_EQUAL_UINT32(0, guard_failures);
    TEST_ASSERT_EQUAL_UINT32(0, held_failures);
}
