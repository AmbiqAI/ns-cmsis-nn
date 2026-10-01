/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

// Blockwise float16 accumulation (AmbiqAI/ns-cmsis-nn#586). Every output of the default entries is checked bit for
// bit against an exact emulation of the kernel's own arithmetic: float16 fused multiply-adds (one rounding each) in
// the kernel's tap order, at most 32 taps per lane per float16 partial, partials widened exactly into float32
// (float32 additions in the kernel's order), one rounding to float16 at the end. The `_acc16` entries are checked
// against the same emulation with no fold (the float16 chain every MVE leg ran before #586), and every output of at
// most 32 taps must agree byte for byte between the two entries. A +0 / -0 difference counts as agreement. The tap
// count is the output's own: the direct kernels skip padded taps, so an edge output counts only its in-range taps;
// the paths that multiply a zero-padded patch (patch-GEMM, the 1xN padded regions, the depthwise to-conv route)
// count every tap of the patch. Under clang (-ffast-math) the float16 reduction of a short dot is the compiler's
// order, so there outputs of at most 32 taps are checked only against the other entry and `_acc16` only for its
// checksum; the fold, whose order the kernels pin, is checked exactly on every compiler.
//
// Two lane shapes are emulated. A lane kernel keeps one output per vector lane and adds one tap per step, the bias
// starting the first partial. A reduction kernel spreads one output's taps over the eight lanes of a vector
// (channel c to lane c % 8, each vector step one tap per lane, or T for the k=3 / k=5 conv1d kernels), counts its
// blocks in vector steps, and when the reduction is longer than 32 taps widens each block's lanes and adds lanes 2j
// and 2j+1 in float32 into pair accumulator j (set by the first block, added to by later ones); the four pair
// accumulators are summed once as (0+1) + (2+3), the bias is added in float32 and the total rounds once. For a single
// block that is ((0+1) + (2+3)) + ((4+5) + (6+7)) in float32. Up to 32 taps the lanes reduce in float16 in the same
// pairing and the bias is added in float16.

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define BA_MVE 1
#else
    #define BA_MVE 0
#endif

#define BA_BLOCK 32
#define BA_NONE INT32_MAX

#if defined(__clang__)
    #define BA_EXACT_SHORT 0
#else
    #define BA_EXACT_SHORT 1
#endif

#define BA_IN_MAX 3072
#define BA_W_MAX 5120
#define BA_OUT_MAX 3072
#define BA_BIAS_MAX 64
#define BA_SCRATCH_MAX 4096

static float16_t ba_in[BA_IN_MAX];
static float16_t ba_w[BA_W_MAX];
static float16_t ba_wp[BA_W_MAX];
static float16_t ba_bias[BA_BIAS_MAX];
static float16_t ba_out[BA_OUT_MAX];
static float16_t ba_out16[BA_OUT_MAX];
static float16_t ba_out_wrap[BA_OUT_MAX];
static float16_t ba_scratch[BA_SCRATCH_MAX];

/* ---------------------------------------------------------------------------------------------------------------- */
/* Exact arithmetic                                                                                                  */
/* ---------------------------------------------------------------------------------------------------------------- */

static uint16_t ba_bits(float16_t v)
{
    uint16_t b;
    memcpy(&b, &v, sizeof(b));
    return b;
}

static float16_t ba_from_bits(uint16_t b)
{
    float16_t v;
    memcpy(&v, &b, sizeof(v));
    return v;
}

// Round a double to the nearest binary16 (ties to even), subnormals included.
static float16_t ba_round_f16(double x)
{
    if (x == 0.0 || isnan(x))
    {
        return (float16_t)x;
    }
    const double ax = fabs(x);
    if (ax >= 65520.0)
    {
        return ba_from_bits(x > 0.0 ? 0x7C00u : 0xFC00u);
    }
    int e;
    (void)frexp(ax, &e);
    int q = e - 11;
    if (q < -24)
    {
        q = -24;
    }
    const double scaled = ldexp(ax, -q);
    double r = floor(scaled);
    const double frac = scaled - r;
    if (frac > 0.5 || (frac == 0.5 && fmod(r, 2.0) != 0.0))
    {
        r += 1.0;
    }
    const double res = ldexp(r, q);
    return (float16_t)(x < 0.0 ? -res : res);
}

// round_f16(a * b + c) with a single rounding. The product is exact in double; the sum is corrected by TwoSum so a
// double rounding cannot land on a binary16 midpoint.
static float16_t ba_fma16(float16_t a, float16_t b, float16_t c)
{
    volatile double p = (double)a * (double)b;
    volatile double cc = (double)c;
    volatile double s = p + cc;
    volatile double z = s - p;
    volatile double err = (p - (s - z)) + (cc - z);
    double r = s;
    if (err != 0.0)
    {
        uint64_t bits;
        memcpy(&bits, &r, sizeof(bits));
        bits = ((err > 0.0) == (r > 0.0)) ? bits + 1u : bits - 1u;
        memcpy(&r, &bits, sizeof(r));
    }
    return ba_round_f16(r);
}

static float16_t ba_add16(float16_t a, float16_t b) { return ba_round_f16((double)a + (double)b); }

static float32_t ba_add32(float32_t a, float32_t b)
{
    volatile float32_t r = a + b;
    return r;
}

// arm_nn_f16_fold_pairs_f32: lanes 2j and 2j+1 widen and add in float32 into pair accumulator j (set on the first
// partial, added to afterwards)
static void ba_fold_pairs_f32(float32_t acc[4], const float16_t p[8], bool first)
{
    for (int32_t j = 0; j < 4; ++j)
    {
        const float32_t pair = ba_add32((float32_t)p[2 * j], (float32_t)p[2 * j + 1]);
        acc[j] = first ? pair : ba_add32(acc[j], pair);
    }
}

// arm_nn_f16_pairs_sum_f32: (0+1) + (2+3)
static float32_t ba_pairs_sum_f32(const float32_t acc[4])
{
    return ba_add32(ba_add32(acc[0], acc[1]), ba_add32(acc[2], acc[3]));
}

// arm_nn_vec_reduce_add_f16
static float16_t ba_sum8_f16(const float16_t p[8])
{
    const float16_t a = ba_add16(p[0], p[1]);
    const float16_t b = ba_add16(p[2], p[3]);
    const float16_t c = ba_add16(p[4], p[5]);
    const float16_t d = ba_add16(p[6], p[7]);
    return ba_add16(ba_add16(a, b), ba_add16(c, d));
}

static float16_t ba_bias_or_zero(const float16_t *bias) { return ba_from_bits(bias ? ba_bits(*bias) : 0u); }

/* Lane kernel: `n` taps in order, bias first, at most `block` taps per float16 partial. */
typedef void (*ba_lane_tap_fn)(const void *ctx, int32_t t, float16_t *x, float16_t *w);

static float16_t ba_emu_lane(const void *ctx, ba_lane_tap_fn fn, int32_t n, const float16_t *bias, int32_t block)
{
    float16_t p = ba_bias_or_zero(bias);
    float32_t acc = 0.0f;
    bool folded = false;
    int32_t in_block = 0;
    for (int32_t t = 0; t < n; ++t)
    {
        if (in_block == block)
        {
            acc = folded ? ba_add32(acc, (float32_t)p) : (float32_t)p;
            folded = true;
            p = ba_from_bits(0u);
            in_block = 0;
        }
        float16_t x;
        float16_t w;
        fn(ctx, t, &x, &w);
        p = ba_fma16(x, w, p);
        ++in_block;
    }
    if (!folded)
    {
        return p;
    }
    return ba_round_f16((double)ba_add32(acc, (float32_t)p));
}

/* Reduction kernel: `n_steps` vector steps of up to `taps` taps per lane; the callback returns false for a lane the
 * step does not reach. `fold` mirrors the kernel's own choice (reduction longer than the block): every block of at
 * most `block_steps` steps is folded into four float32 pair accumulators, which are summed once; then the bias is
 * added in float32 and the total rounds once. */
typedef bool (*ba_red_tap_fn)(const void *ctx, int32_t step, int32_t lane, int32_t t, float16_t *x, float16_t *w);

static float16_t ba_emu_reduce(const void *ctx,
                               ba_red_tap_fn fn,
                               int32_t n_steps,
                               int32_t taps,
                               const float16_t *bias,
                               bool fold,
                               int32_t block_steps)
{
    float16_t p[8];
    for (int32_t l = 0; l < 8; ++l)
    {
        p[l] = ba_from_bits(0u);
    }
    float32_t pairs[4];
    bool first = true;
    int32_t in_block = 0;
    for (int32_t s = 0; s < n_steps; ++s)
    {
        if (fold && in_block == block_steps)
        {
            ba_fold_pairs_f32(pairs, p, first);
            first = false;
            for (int32_t l = 0; l < 8; ++l)
            {
                p[l] = ba_from_bits(0u);
            }
            in_block = 0;
        }
        for (int32_t l = 0; l < 8; ++l)
        {
            for (int32_t t = 0; t < taps; ++t)
            {
                float16_t x;
                float16_t w;
                if (fn(ctx, s, l, t, &x, &w))
                {
                    p[l] = ba_fma16(x, w, p[l]);
                }
            }
        }
        ++in_block;
    }
    if (fold)
    {
        ba_fold_pairs_f32(pairs, p, first);
        const float32_t b32 = bias ? (float32_t)*bias : 0.0f;
        return ba_round_f16((double)ba_add32(b32, ba_pairs_sum_f32(pairs)));
    }
    return ba_add16(ba_bias_or_zero(bias), ba_sum8_f16(p));
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Data and checking                                                                                                 */
/* ---------------------------------------------------------------------------------------------------------------- */

static uint32_t ba_hash(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u + seed * 40503u + 0x9E3779B9u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    return h;
}

// Full-precision values in [-1, 1): products carry 22 significant bits, so every float16 addition rounds.
static void ba_fill(float16_t *dst, int32_t n, uint32_t seed)
{
    for (int32_t i = 0; i < n; ++i)
    {
        dst[i] = (float16_t)((float32_t)((int32_t)(ba_hash((uint32_t)i, seed) & 0xFFFFu) - 32768) / 32768.0f);
    }
}

// NT_N_PACKED: [ceil(n / 8)][k][8], tail lanes zero.
static void ba_pack_nt_n(const float16_t *w, float16_t *wp, int32_t n, int32_t k)
{
    const int32_t blocks = (n + 7) / 8;
    for (int32_t b = 0; b < blocks; ++b)
    {
        for (int32_t kk = 0; kk < k; ++kk)
        {
            for (int32_t l = 0; l < 8; ++l)
            {
                const int32_t col = b * 8 + l;
                wp[((size_t)b * k + kk) * 8 + l] = ba_from_bits(col < n ? ba_bits(w[(size_t)col * k + kk]) : 0u);
            }
        }
    }
}

typedef float16_t (*ba_ref_fn)(const void *ctx, int32_t idx, int32_t block);

/* Taps summed into output `idx`; NULL where every output of the case has the same count. */
typedef int32_t (*ba_taps_fn)(const void *ctx, int32_t idx);

static int32_t ba_taps_of(ba_taps_fn taps, const void *ctx, int32_t idx, int32_t k_taps)
{
    return taps ? taps(ctx, idx) : k_taps;
}

static uint32_t ba_checksum;

/* When set, the emulated outputs are clamped to [ba_clamp_lo, ba_clamp_hi] after their rounding, as the kernels do. */
static bool ba_clamp_on;
static float16_t ba_clamp_lo;
static float16_t ba_clamp_hi;

static uint16_t ba_clamp_bits(uint16_t b)
{
    const double v = (double)ba_from_bits(b);
    if (v > (double)ba_clamp_hi)
    {
        return ba_bits(ba_clamp_hi);
    }
    if (v < (double)ba_clamp_lo)
    {
        return ba_bits(ba_clamp_lo);
    }
    return b;
}

// Neighbouring float16 values (finite, nonzero inputs).
static uint16_t ba_next_up(uint16_t b) { return (b & 0x8000u) ? (uint16_t)(b - 1u) : (uint16_t)(b + 1u); }

static uint16_t ba_next_down(uint16_t b) { return (b & 0x8000u) ? (uint16_t)(b + 1u) : (uint16_t)(b - 1u); }
static int32_t ba_failures; /* cases of the current test that disagree; every case runs before the test asserts */

static void ba_begin(void) { ba_failures = 0; }

static void ba_end(void) { TEST_ASSERT_EQUAL_INT32(0, ba_failures); }

// Compares n outputs with the emulation at `block` and returns the mismatch count, printing the first few. Every
// output enters the checksum; outputs of at most 32 taps are compared only when `check_short`, longer ones only when
// `check_long`.
static int32_t ba_check(const char *what,
                        const float16_t *out,
                        int32_t n,
                        ba_ref_fn ref,
                        const void *ctx,
                        int32_t block,
                        ba_taps_fn taps,
                        int32_t k_taps,
                        bool check_short,
                        bool check_long)
{
    int32_t bad = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        const uint16_t got = ba_bits(out[i]);
        ba_checksum = (ba_checksum ^ got) * 16777619u;
        if (!((ba_taps_of(taps, ctx, i, k_taps) <= BA_BLOCK) ? check_short : check_long))
        {
            continue;
        }
        uint16_t want = ba_bits(ref(ctx, i, block));
        if (ba_clamp_on)
        {
            want = ba_clamp_bits(want);
        }
        if (got != want && !(((got | want) & 0x7FFFu) == 0u))
        {
            if (bad < 3)
            {
                printf("%s[%ld]: got 0x%04x want 0x%04x\n", what, (long)i, (unsigned)got, (unsigned)want);
            }
            ++bad;
        }
    }
    return bad;
}

static int32_t ba_count_diff(const float16_t *a, const float16_t *b, int32_t n)
{
    int32_t diff = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        diff += ba_bits(a[i]) != ba_bits(b[i]);
    }
    return diff;
}

// Outputs of at most 32 taps where the two entries differ; *n_short counts those outputs.
static int32_t ba_count_short_diff(ba_taps_fn taps, const void *ctx, int32_t k_taps, int32_t n, int32_t *n_short)
{
    int32_t diff = 0;
    *n_short = 0;
    for (int32_t i = 0; i < n; ++i)
    {
        if (ba_taps_of(taps, ctx, i, k_taps) <= BA_BLOCK)
        {
            ++*n_short;
            diff += ba_bits(ba_out[i]) != ba_bits(ba_out16[i]);
        }
    }
    return diff;
}

static void ba_expect_same(const char *what, int32_t n);

// Default entry against the fold emulation and _acc16 against the float16 chain; every output of at most 32 taps
// (its own count, from `taps`, or `k_taps` for all of them) must also agree byte for byte between the entries.
// Scalar legs (non-MVE builds) accumulate in float32 on both entries, which must then agree everywhere.
static void ba_expect_taps(const char *what, int32_t n, ba_ref_fn ref, const void *ctx, int32_t k_taps, ba_taps_fn taps)
{
    char label[80];
    if (!BA_MVE)
    {
        ba_expect_same(what, n);
        return;
    }
    ba_checksum = 2166136261u;
    snprintf(label, sizeof(label), "%s acc16", what);
    const int32_t bad16 = ba_check(label, ba_out16, n, ref, ctx, BA_NONE, taps, k_taps, BA_EXACT_SHORT, BA_EXACT_SHORT);
    const uint32_t sum16 = ba_checksum;
    snprintf(label, sizeof(label), "%s fold", what);
    const int32_t bad = ba_check(label, ba_out, n, ref, ctx, BA_BLOCK, taps, k_taps, BA_EXACT_SHORT, true);
    int32_t n_short = 0;
    const int32_t same = ba_count_short_diff(taps, ctx, k_taps, n, &n_short);
    printf("CASE %s n=%ld fold_mismatch=%ld acc16_mismatch=%ld short_k_diff=%ld acc16_cksum=%08lx short_n=%ld\n",
           what,
           (long)n,
           (long)bad,
           (long)bad16,
           (long)same,
           (unsigned long)sum16,
           (long)n_short);
    ba_failures += (bad != 0) + (bad16 != 0) + (same != 0);
}

static void ba_expect(const char *what, int32_t n, ba_ref_fn ref, const void *ctx, int32_t k_taps)
{
    ba_expect_taps(what, n, ref, ctx, k_taps, NULL);
}

// Shapes whose MVE route is the gather kernel with lanes as outputs and fewer than 32 taps: nothing folds, so the
// two entries must agree byte for byte.
static void ba_expect_same(const char *what, int32_t n)
{
    const int32_t diff = ba_count_diff(ba_out, ba_out16, n);
    ba_checksum = 2166136261u;
    for (int32_t i = 0; i < n; ++i)
    {
        ba_checksum = (ba_checksum ^ ba_bits(ba_out16[i])) * 16777619u;
    }
    printf("CASE %s n=%ld entries_diff=%ld acc16_cksum=%08lx\n", what, (long)n, (long)diff, (unsigned long)ba_checksum);
    ba_failures += diff != 0;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Fully connected (arm_nn_mat_mult_nt_t_f16 / _nt_n_packed_f16)                                                     */
/* ---------------------------------------------------------------------------------------------------------------- */

typedef struct
{
    int32_t batch;
    int32_t k;
    int32_t n;
    const float16_t *bias;
    int32_t out; /* output being emulated */
} ba_fc;

static bool ba_fc_red_tap(const void *vctx, int32_t s, int32_t l, int32_t t, float16_t *x, float16_t *w)
{
    (void)t;
    const ba_fc *c = (const ba_fc *)vctx;
    const int32_t kk = s * 8 + l;
    if (kk >= c->k)
    {
        return false;
    }
    const int32_t b = c->out / c->n;
    const int32_t col = c->out % c->n;
    *x = ba_in[(size_t)b * c->k + kk];
    *w = ba_w[(size_t)col * c->k + kk];
    return true;
}

static void ba_fc_lane_tap(const void *vctx, int32_t t, float16_t *x, float16_t *w)
{
    const ba_fc *c = (const ba_fc *)vctx;
    const int32_t b = c->out / c->n;
    const int32_t col = c->out % c->n;
    *x = ba_in[(size_t)b * c->k + t];
    *w = ba_w[(size_t)col * c->k + t];
}

static float16_t ba_fc_ref_std(const void *vctx, int32_t idx, int32_t block)
{
    ba_fc c = *(const ba_fc *)vctx;
    c.out = idx;
    const float16_t *bias = c.bias ? &c.bias[idx % c.n] : NULL;
    /* K >= 32: every row takes the contiguous-K groups or the remainder dot, both reductions. */
    return ba_emu_reduce(&c, ba_fc_red_tap, (c.k + 7) / 8, 1, bias, c.k > block, block);
}

static float16_t ba_fc_ref_packed(const void *vctx, int32_t idx, int32_t block)
{
    ba_fc c = *(const ba_fc *)vctx;
    c.out = idx;
    return ba_emu_lane(&c, ba_fc_lane_tap, c.k, c.bias ? &c.bias[idx % c.n] : NULL, block);
}

/* When set, ba_fc_case replaces its random data: every output's first block puts +2048 on lane 0 and 2^-13 on lane 2,
 * its second block puts -2048 on lane 0. Summing each block's lanes on its own loses the 2^-13 in float32; the pair
 * accumulators cancel lane 0 across blocks first and keep it, so the result tells the two orders apart. */
static bool ba_fc_crafted;

/* When set, ba_fc_case narrows the activation range to one float16 step inside the folded results of outputs 0 and
 * 1, so that those two outputs clamp right next to their single rounding. */
static bool ba_fc_clamp;

static void ba_fc_case(int32_t batch, int32_t k, int32_t n, bool packed, bool bias)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {batch, 1, 1, k};
    const cmsis_nn_dims filter_dims = {k, 1, 1, n};
    const cmsis_nn_dims bias_dims = {1, 1, 1, n};
    const cmsis_nn_dims output_dims = {batch, 1, 1, n};
    cmsis_nn_fc_params_f16 fc_params;
    char what[64];

    TEST_ASSERT_TRUE(batch * k <= BA_IN_MAX && n * k <= BA_W_MAX && (!packed || ((n + 7) / 8) * 8 * k <= BA_W_MAX));
    memset(&fc_params, 0, sizeof(fc_params));
    fc_params.activation.min = (float16_t)-6.0e4f;
    fc_params.activation.max = (float16_t)6.0e4f;
    fc_params.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
    ba_fill(ba_in, batch * k, (uint32_t)k * 3u + 1u);
    ba_fill(ba_w, n * k, (uint32_t)k * 3u + 2u);
    ba_fill(ba_bias, n, (uint32_t)k * 3u + 3u);
    if (ba_fc_crafted)
    {
        TEST_ASSERT_TRUE(k > 256 + 8);
        memset(ba_in, 0, sizeof(float16_t) * (size_t)(batch * k));
        memset(ba_w, 0, sizeof(float16_t) * (size_t)(n * k));
        for (int32_t b = 0; b < batch; ++b)
        {
            ba_in[b * k + 0] = (float16_t)1.0f;
            ba_in[b * k + 2] = (float16_t)1.0f;
            ba_in[b * k + 256] = (float16_t)1.0f;
        }
        for (int32_t j = 0; j < n; ++j)
        {
            ba_w[j * k + 0] = (float16_t)2048.0f;
            ba_w[j * k + 2] = (float16_t)1.220703125e-4f; /* 2^-13 */
            ba_w[j * k + 256] = (float16_t)-2048.0f;
        }
    }
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, n, k);
        w = ba_wp;
    }
    const float16_t *b = bias ? ba_bias : NULL;
    const ba_fc c = {batch, k, n, b, 0};
    if (ba_fc_clamp)
    {
        const ba_ref_fn ref = packed ? ba_fc_ref_packed : ba_fc_ref_std;
        const float16_t r0 = ref(&c, 0, BA_BLOCK);
        const float16_t r1 = ref(&c, 1, BA_BLOCK);
        const bool up = (double)r0 > (double)r1;
        ba_clamp_hi = ba_from_bits(ba_next_down(ba_bits(up ? r0 : r1)));
        ba_clamp_lo = ba_from_bits(ba_next_up(ba_bits(up ? r1 : r0)));
        TEST_ASSERT_TRUE(k > BA_BLOCK && (double)ba_clamp_lo < (double)ba_clamp_hi);
        fc_params.activation.min = ba_clamp_lo;
        fc_params.activation.max = ba_clamp_hi;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f16(&ctx,
                                              &fc_params,
                                              &input_dims,
                                              ba_in,
                                              &filter_dims,
                                              w,
                                              &bias_dims,
                                              b,
                                              &output_dims,
                                              ba_out,
                                              ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f16_acc16(&ctx,
                                                    &fc_params,
                                                    &input_dims,
                                                    ba_in,
                                                    &filter_dims,
                                                    w,
                                                    &bias_dims,
                                                    b,
                                                    &output_dims,
                                                    ba_out16,
                                                    ARM_NN_LAYOUT_NHWC));
    snprintf(
        what, sizeof(what), "fc%s%s k%ld n%ld", packed ? "-packed" : "", ba_fc_clamp ? "-clamp" : "", (long)k, (long)n);
    if (!packed && k < 32)
    {
        /* Below the contiguous-K threshold the gather kernel's lanes are outputs; nothing folds there. */
        ba_expect_same(what, batch * n);
        return;
    }
    ba_clamp_on = ba_fc_clamp;
    ba_expect(what, batch * n, packed ? ba_fc_ref_packed : ba_fc_ref_std, &c, k);
    ba_clamp_on = false;
}

void ba_fc_std_long_k(void)
{
    ba_begin();
    ba_fc_case(2, 33, 5, false, true);
    ba_fc_case(1, 64, 5, false, true);
    ba_fc_case(2, 98, 5, false, false);
    ba_fc_case(1, 280, 5, false, true);
    ba_fc_case(2, 1024, 5, false, true);
    ba_end();
}

void ba_fc_packed_long_k(void)
{
    ba_begin();
    ba_fc_case(2, 33, 5, true, true);
    ba_fc_case(1, 64, 9, true, true);
    ba_fc_case(2, 98, 5, true, false);
    ba_fc_case(1, 280, 13, true, true);
    ba_fc_case(1, 640, 5, true, true);
    ba_end();
}

void ba_fc_block_order(void)
{
    ba_begin();
    ba_fc_crafted = true;
    ba_fc_case(1, 512, 5, false, false);
    ba_fc_crafted = false;
    ba_end();
}

void ba_fc_clamp_edge(void)
{
    ba_begin();
    ba_fc_clamp = true;
    ba_fc_case(2, 72, 5, false, true);
    ba_fc_case(2, 300, 5, false, true);
    ba_fc_case(2, 72, 5, true, true);
    ba_fc_clamp = false;
    ba_end();
}

void ba_fc_short_k(void)
{
    ba_begin();
    ba_fc_case(2, 8, 13, false, true);
    ba_fc_case(1, 31, 9, false, true);
    ba_fc_case(2, 32, 5, false, true);
    ba_fc_case(1, 32, 5, false, false);
    ba_fc_case(2, 17, 5, true, true);
    ba_fc_case(1, 32, 13, true, true);
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* 1x1 convolution (matmul-backed)                                                                                   */
/* ---------------------------------------------------------------------------------------------------------------- */

static void ba_conv1x1_case(int32_t k, int32_t n, bool packed)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {1, 1, 3, k};
    const cmsis_nn_dims filter_dims = {n, 1, 1, k};
    const cmsis_nn_dims bias_dims = {1, 1, 1, n};
    const cmsis_nn_dims output_dims = {1, 1, 3, n};
    cmsis_nn_conv_params_f16 p;
    char what[64];

    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.dilation.h = 1;
    p.dilation.w = 1;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
    TEST_ASSERT_TRUE(3 * k <= BA_IN_MAX && ((n + 7) / 8) * 8 * k <= BA_W_MAX);
    ba_fill(ba_in, 3 * k, (uint32_t)k + 11u);
    ba_fill(ba_w, n * k, (uint32_t)k + 12u);
    ba_fill(ba_bias, n, (uint32_t)k + 13u);
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, n, k);
        w = ba_wp;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_1x1_f16(&ctx,
                                           &p,
                                           &input_dims,
                                           ba_in,
                                           &filter_dims,
                                           w,
                                           &bias_dims,
                                           ba_bias,
                                           &output_dims,
                                           ba_out,
                                           ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_1x1_f16_acc16(&ctx,
                                                 &p,
                                                 &input_dims,
                                                 ba_in,
                                                 &filter_dims,
                                                 w,
                                                 &bias_dims,
                                                 ba_bias,
                                                 &output_dims,
                                                 ba_out16,
                                                 ARM_NN_LAYOUT_NHWC));
    const ba_fc c = {3, k, n, ba_bias, 0};
    snprintf(what, sizeof(what), "1x1%s k%ld", packed ? "-packed" : "", (long)k);
    ba_expect(what, 3 * n, packed ? ba_fc_ref_packed : ba_fc_ref_std, &c, k);
}

void ba_conv1x1_long_k(void)
{
    ba_begin();
    ba_conv1x1_case(64, 5, false);
    ba_conv1x1_case(98, 9, false);
    ba_conv1x1_case(280, 5, true);
    ba_conv1x1_case(32, 5, false);
    ba_conv1x1_case(24, 9, true);
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Batch matmul, no adjoint: arm_nn_mat_mult_nt_t_f16 / _nt_n_packed_f16 without a bias. It has no `_acc16` entry;  */
/* the float16-lane side is the matmul's own `_acc16` entry on the same operands.                                   */
/* ---------------------------------------------------------------------------------------------------------------- */

static void ba_bmm_case(int32_t rows, int32_t k, int32_t n, bool packed)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims lhs_dims = {1, 1, k, rows};
    const cmsis_nn_dims rhs_dims = {1, 1, k, n};
    const cmsis_nn_dims output_dims = {1, 1, n, rows};
    const cmsis_nn_bmm_params_f16 p = {.adj_x = false,
                                       .adj_y = false,
                                       .activation = {(float16_t)-6.0e4f, (float16_t)6.0e4f},
                                       .rhs_format =
                                           packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD};
    char what[64];

    TEST_ASSERT_TRUE(rows * k <= BA_IN_MAX && ((n + 7) / 8) * 8 * k <= BA_W_MAX && rows * n <= BA_OUT_MAX);
    ba_fill(ba_in, rows * k, (uint32_t)k + 71u);
    ba_fill(ba_w, n * k, (uint32_t)k + 72u);
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, n, k);
        w = ba_wp;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_batch_matmul_f16(&ctx, &p, &lhs_dims, ba_in, &rhs_dims, w, &output_dims, ba_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      (packed ? arm_nn_mat_mult_nt_n_packed_f16_acc16 : arm_nn_mat_mult_nt_t_f16_acc16)(
                          ba_in, w, NULL, ba_out16, rows, n, k, n, p.activation.min, p.activation.max));
    const ba_fc c = {rows, k, n, NULL, 0};
    snprintf(what, sizeof(what), "bmm%s k%ld n%ld", packed ? "-packed" : "", (long)k, (long)n);
    ba_expect(what, rows * n, packed ? ba_fc_ref_packed : ba_fc_ref_std, &c, k);
}

void ba_batch_matmul_long_k(void)
{
    ba_begin();
    ba_bmm_case(3, 72, 5, false);
    ba_bmm_case(2, 300, 9, false);
    ba_bmm_case(3, 72, 5, true);
    ba_bmm_case(3, 32, 5, false);
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* 1xN convolution: padded regions through the matmul, the no-padding region through the strided kernel            */
/* ---------------------------------------------------------------------------------------------------------------- */

typedef struct
{
    int32_t in_w;
    int32_t in_c;
    int32_t kw;
    int32_t pad;
    int32_t out_c;
    int32_t out_w;
    int32_t dil;
    bool packed;
    int32_t ox;
    int32_t oc;
} ba_c1d;

// Patch element j (tap-major, channel-minor) of output x; zero where the tap falls in the padding.
static float16_t ba_c1d_patch(const ba_c1d *c, int32_t j)
{
    const int32_t tap = j / c->in_c;
    const int32_t ch = j % c->in_c;
    const int32_t ix = c->ox - c->pad + tap * c->dil;
    return ba_from_bits((ix >= 0 && ix < c->in_w) ? ba_bits(ba_in[(size_t)ix * c->in_c + ch]) : 0u);
}

static bool ba_c1d_red_tap(const void *vctx, int32_t s, int32_t l, int32_t t, float16_t *x, float16_t *w)
{
    (void)t;
    const ba_c1d *c = (const ba_c1d *)vctx;
    const int32_t k = c->kw * c->in_c;
    const int32_t j = s * 8 + l;
    if (j >= k)
    {
        return false;
    }
    *x = ba_c1d_patch(c, j);
    *w = ba_w[(size_t)c->oc * k + j];
    return true;
}

static void ba_c1d_lane_tap(const void *vctx, int32_t t, float16_t *x, float16_t *w)
{
    const ba_c1d *c = (const ba_c1d *)vctx;
    *x = ba_c1d_patch(c, t);
    *w = ba_w[(size_t)c->oc * c->kw * c->in_c + t];
}

static float16_t ba_c1xn_ref(const void *vctx, int32_t idx, int32_t block)
{
    ba_c1d c = *(const ba_c1d *)vctx;
    c.ox = idx / c.out_c;
    c.oc = idx % c.out_c;
    const int32_t k = c.kw * c.in_c;
    const float16_t *bias = &ba_bias[c.oc];
    const int32_t base = c.ox - c.pad;
    const bool interior = base >= 0 && base + c.kw <= c.in_w;
    if (c.packed)
    {
        return ba_emu_lane(&c, ba_c1d_lane_tap, k, bias, block);
    }
    /* Strided kernel (no-padding region): gather lanes for whole blocks of 8, then 4, of output channels, then the
     * remainder dot. Padded regions: the matmul's contiguous-K groups of four and its remainder dot. */
    const int32_t lane_oc = (c.out_c / 8) * 8 + ((c.out_c % 8) / 4) * 4;
    if (interior && c.oc < lane_oc)
    {
        return ba_emu_lane(&c, ba_c1d_lane_tap, k, bias, block);
    }
    return ba_emu_reduce(&c, ba_c1d_red_tap, (k + 7) / 8, 1, bias, k > block, block);
}

static void ba_c1xn_case(int32_t in_w, int32_t in_c, int32_t kw, int32_t out_c, bool packed)
{
    const int32_t pad = kw / 2;
    const cmsis_nn_dims input_dims = {1, 1, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, kw, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, 1, in_w, out_c};
    cmsis_nn_conv_params_f16 p;
    char what[64];

    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.padding.w = pad;
    p.dilation.h = 1;
    p.dilation.w = 1;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
    const int32_t buf =
        arm_convolve_1_x_n_f16_get_buffer_size(&p, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
    TEST_ASSERT_TRUE(buf > 0 && buf <= (int32_t)sizeof(ba_scratch));
    const cmsis_nn_context ctx = {ba_scratch, buf};
    TEST_ASSERT_TRUE(in_w * in_c <= BA_IN_MAX && ((out_c + 7) / 8) * 8 * kw * in_c <= BA_W_MAX);
    ba_fill(ba_in, in_w * in_c, (uint32_t)in_c + 21u);
    ba_fill(ba_w, out_c * kw * in_c, (uint32_t)in_c + 22u);
    ba_fill(ba_bias, out_c, (uint32_t)in_c + 23u);
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, out_c, kw * in_c);
        w = ba_wp;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_1_x_n_f16(&ctx,
                                             &p,
                                             &input_dims,
                                             ba_in,
                                             &filter_dims,
                                             w,
                                             &bias_dims,
                                             ba_bias,
                                             &output_dims,
                                             ba_out,
                                             ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_1_x_n_f16_acc16(&ctx,
                                                   &p,
                                                   &input_dims,
                                                   ba_in,
                                                   &filter_dims,
                                                   w,
                                                   &bias_dims,
                                                   ba_bias,
                                                   &output_dims,
                                                   ba_out16,
                                                   ARM_NN_LAYOUT_NHWC));
    const ba_c1d c = {in_w, in_c, kw, pad, out_c, in_w, 1, packed, 0, 0};
    snprintf(what, sizeof(what), "1xN%s k%ld c%ld", packed ? "-packed" : "", (long)kw, (long)in_c);
    if (!packed && kw * in_c < 32)
    {
        /* Short padded rows take the matmul's gather lanes; nothing folds there. */
        ba_expect_same(what, in_w * out_c);
        return;
    }
    ba_expect(what, in_w * out_c, ba_c1xn_ref, &c, kw * in_c);
}

void ba_conv_1xn_long_k(void)
{
    ba_begin();
    ba_c1xn_case(16, 14, 7, 13, false); /* K = 98 */
    ba_c1xn_case(12, 40, 7, 13, false); /* K = 280 */
    ba_c1xn_case(12, 40, 7, 5, true);
    ba_c1xn_case(10, 5, 7, 13, false);  /* K = 35 */
    ba_c1xn_case(10, 4, 7, 13, false);  /* K = 28 */
    ba_c1xn_case(12, 14, 7, 21, false); /* 16- and 4-lane gather groups, then the remainder dot */
    /* K = 56. The padded outputs multiply a zero-padded patch row, so they count all 56 taps, 32 of them in range at
     * the ends. */
    ba_c1xn_case(10, 8, 7, 13, false);
    ba_c1xn_case(10, 8, 7, 13, true);
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Dilated conv1d through the generic convolution's patch-GEMM                                                       */
/* ---------------------------------------------------------------------------------------------------------------- */

static float16_t ba_dil_ref(const void *vctx, int32_t idx, int32_t block)
{
    ba_c1d c = *(const ba_c1d *)vctx;
    c.ox = idx / c.out_c;
    c.oc = idx % c.out_c;
    const int32_t k = c.kw * c.in_c;
    if (c.packed)
    {
        return ba_emu_lane(&c, ba_c1d_lane_tap, k, &ba_bias[c.oc], block);
    }
    return ba_emu_reduce(&c, ba_c1d_red_tap, (k + 7) / 8, 1, &ba_bias[c.oc], k > block, block);
}

static void ba_dil_case(int32_t in_w, int32_t in_c, int32_t out_c, bool packed)
{
    const int32_t kw = 3;
    const int32_t dil = 2;
    const cmsis_nn_dims input_dims = {1, 1, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, kw, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, 1, in_w, out_c};
    cmsis_nn_conv_params_f16 p;
    char what[64];

    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.padding.w = dil;
    p.dilation.h = 1;
    p.dilation.w = dil;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
    const cmsis_nn_context ctx = {ba_scratch, (int32_t)sizeof(ba_scratch)};
    TEST_ASSERT_TRUE(in_w * in_c <= BA_IN_MAX && ((out_c + 7) / 8) * 8 * kw * in_c <= BA_W_MAX);
    ba_fill(ba_in, in_w * in_c, (uint32_t)in_c + 31u);
    ba_fill(ba_w, out_c * kw * in_c, (uint32_t)in_c + 32u);
    ba_fill(ba_bias, out_c, (uint32_t)in_c + 33u);
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, out_c, kw * in_c);
        w = ba_wp;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f16(&ctx,
                                       &p,
                                       &input_dims,
                                       ba_in,
                                       &filter_dims,
                                       w,
                                       &bias_dims,
                                       ba_bias,
                                       &output_dims,
                                       ba_out,
                                       ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f16_acc16(&ctx,
                                             &p,
                                             &input_dims,
                                             ba_in,
                                             &filter_dims,
                                             w,
                                             &bias_dims,
                                             ba_bias,
                                             &output_dims,
                                             ba_out16,
                                             ARM_NN_LAYOUT_NHWC));
    const ba_c1d c = {in_w, in_c, kw, dil, out_c, in_w, dil, packed, 0, 0};
    snprintf(what, sizeof(what), "dil%s c%ld", packed ? "-packed" : "", (long)in_c);
    if (!packed && kw * in_c < 32)
    {
        ba_expect_same(what, in_w * out_c);
        return;
    }
    ba_expect(what, in_w * out_c, ba_dil_ref, &c, kw * in_c);
}

void ba_conv1d_dilated_long_k(void)
{
    ba_begin();
    ba_dil_case(10, 11, 9, false); /* K = 33 */
    ba_dil_case(10, 33, 9, false); /* K = 99 */
    ba_dil_case(8, 94, 9, false);  /* K = 282 */
    ba_dil_case(10, 33, 9, true);
    ba_dil_case(10, 10, 9, false); /* K = 30 */
    ba_dil_case(10, 16, 9, false); /* K = 48; the patch-GEMM rows count their zero taps, 32 in range at the ends */
    ba_dil_case(10, 16, 9, true);
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* k=3 / k=5 conv1d specializations                                                                                  */
/* ---------------------------------------------------------------------------------------------------------------- */

static bool ba_spec_red_tap(const void *vctx, int32_t s, int32_t l, int32_t t, float16_t *x, float16_t *w)
{
    const ba_c1d *c = (const ba_c1d *)vctx;
    const int32_t ch = s * 8 + l;
    if (ch >= c->in_c)
    {
        return false;
    }
    *x = ba_in[(size_t)(c->ox + t) * c->in_c + ch];
    *w = ba_w[((size_t)c->oc * c->kw + t) * c->in_c + ch];
    return true;
}

static void ba_spec_lane_tap(const void *vctx, int32_t t, float16_t *x, float16_t *w)
{
    const ba_c1d *c = (const ba_c1d *)vctx;
    const int32_t ch = t / c->kw;
    const int32_t tap = t % c->kw;
    *x = ba_in[(size_t)(c->ox + tap) * c->in_c + ch];
    *w = ba_w[((size_t)c->oc * c->kw + tap) * c->in_c + ch];
}

static float16_t ba_spec_ref(const void *vctx, int32_t idx, int32_t block)
{
    ba_c1d c = *(const ba_c1d *)vctx;
    c.ox = idx / c.out_c;
    c.oc = idx % c.out_c;
    const int32_t taps = c.kw * c.in_c;
    /* A partial covers block / kw channels (per vector step for the OHWI kernel, per channel for the packed one). */
    const int32_t per = (block == BA_NONE) ? BA_NONE : (block / c.kw);
    if (c.packed)
    {
        return ba_emu_lane(&c, ba_spec_lane_tap, taps, &ba_bias[c.oc], (block == BA_NONE) ? BA_NONE : per * c.kw);
    }
    return ba_emu_reduce(&c, ba_spec_red_tap, (c.in_c + 7) / 8, c.kw, &ba_bias[c.oc], taps > block, per);
}

static void ba_spec_case(int32_t kw, int32_t in_w, int32_t in_c, int32_t out_c, bool packed)
{
    const cmsis_nn_dims input_dims = {1, 1, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, kw, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const int32_t out_w = in_w - kw + 1;
    const cmsis_nn_dims output_dims = {1, 1, out_w, out_c};
    cmsis_nn_conv_params_f16 p;
    char what[64];

    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.dilation.h = 1;
    p.dilation.w = 1;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
    const cmsis_nn_context ctx = {ba_scratch, (int32_t)sizeof(ba_scratch)};
    TEST_ASSERT_TRUE(in_w * in_c <= BA_IN_MAX && ((out_c + 7) / 8) * 8 * kw * in_c <= BA_W_MAX);
    ba_fill(ba_in, in_w * in_c, (uint32_t)(in_c * kw) + 41u);
    ba_fill(ba_w, out_c * kw * in_c, (uint32_t)(in_c * kw) + 42u);
    ba_fill(ba_bias, out_c, (uint32_t)(in_c * kw) + 43u);
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, out_c, kw * in_c);
        w = ba_wp;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f16(&ctx,
                                       &p,
                                       &input_dims,
                                       ba_in,
                                       &filter_dims,
                                       w,
                                       &bias_dims,
                                       ba_bias,
                                       &output_dims,
                                       ba_out,
                                       ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f16_acc16(&ctx,
                                             &p,
                                             &input_dims,
                                             ba_in,
                                             &filter_dims,
                                             w,
                                             &bias_dims,
                                             ba_bias,
                                             &output_dims,
                                             ba_out16,
                                             ARM_NN_LAYOUT_NHWC));
    const ba_c1d c = {in_w, in_c, kw, 0, out_c, out_w, 1, packed, 0, 0};
    snprintf(what, sizeof(what), "k%ld%s c%ld", (long)kw, packed ? "-packed" : "", (long)in_c);
    ba_expect(what, out_w * out_c, ba_spec_ref, &c, kw * in_c);
}

void ba_conv1d_spec_long_k(void)
{
    ba_begin();
    ba_spec_case(3, 6, 11, 5, false); /* K = 33 */
    ba_spec_case(3, 5, 33, 5, false); /* K = 99 */
    ba_spec_case(3, 4, 94, 5, false); /* K = 282 */
    ba_spec_case(3, 5, 33, 13, true);
    ba_spec_case(3, 5, 10, 5, false);  /* K = 30 */
    ba_spec_case(5, 7, 24, 5, false);  /* K = 120 */
    ba_spec_case(5, 7, 16, 17, false); /* the in_c == 16 kernel, K = 80 */
    ba_spec_case(5, 7, 56, 5, false);  /* K = 280 */
    ba_spec_case(5, 7, 20, 9, true);
    ba_spec_case(5, 7, 20, 5, false); /* in_c 20 = 16 + a 4-channel tail (#588) */
    ba_spec_case(5, 7, 6, 5, false); /* K = 30 */
    ba_end();
}

static uint32_t ba_out_fnv(const float16_t *out, int32_t n)
{
    uint32_t h = 2166136261u;
    for (int32_t i = 0; i < n; ++i)
    {
        h = (h ^ ba_bits(out[i])) * 16777619u;
    }
    return h;
}

/* k5 output channels past the last whole group of four, alone (out_c < 4) and after one (out_c 7), at depths with
 * only a partial 8-channel step, only whole steps, both, several fold spans, and depths ending exactly on a fold span
 * (48, 96), where no partial step runs. Each case prints a hash of both entries' outputs so two builds can be
 * compared bit for bit. */
void ba_conv1d_spec_k5_tail(void)
{
    static const int32_t in_cs[] = {1, 7, 8, 9, 15, 17, 23, 41, 48, 57, 96, 100};
    static const int32_t out_cs[] = {1, 3, 7};
    const int32_t in_w = 6;
    ba_begin();
    for (size_t i = 0; i < sizeof(in_cs) / sizeof(in_cs[0]); ++i)
    {
        for (size_t j = 0; j < sizeof(out_cs) / sizeof(out_cs[0]); ++j)
        {
            ba_spec_case(5, in_w, in_cs[i], out_cs[j], false);
            const int32_t n = (in_w - 4) * out_cs[j];
            printf("K5TAIL c%ld oc%ld out=%08lx out16=%08lx\n",
                   (long)in_cs[i],
                   (long)out_cs[j],
                   (unsigned long)ba_out_fnv(ba_out, n),
                   (unsigned long)ba_out_fnv(ba_out16, n));
        }
    }
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Direct OHWI / NT_N_PACKED fallback (no scratch)                                                                   */
/* ---------------------------------------------------------------------------------------------------------------- */

typedef struct
{
    int32_t hw;     /* input height and width */
    int32_t out_hw; /* output height and width */
    int32_t pad;
    int32_t dil;
    int32_t in_c;
    int32_t out_c;
    int32_t pos[9]; /* in-range taps of the output being emulated: input pixel index */
    int32_t tap[9]; /* ... and filter tap index */
    int32_t n_pos;
    int32_t oc;
} ba_direct;

static bool ba_direct_red_tap(const void *vctx, int32_t s, int32_t l, int32_t t, float16_t *x, float16_t *w)
{
    (void)t;
    const ba_direct *c = (const ba_direct *)vctx;
    const int32_t steps = (c->in_c + 7) / 8;
    const int32_t pi = s / steps;
    const int32_t ch = (s % steps) * 8 + l;
    if (ch >= c->in_c)
    {
        return false;
    }
    *x = ba_in[(size_t)c->pos[pi] * c->in_c + ch];
    *w = ba_w[((size_t)c->oc * 9 + c->tap[pi]) * c->in_c + ch];
    return true;
}

static void ba_direct_lane_tap(const void *vctx, int32_t t, float16_t *x, float16_t *w)
{
    const ba_direct *c = (const ba_direct *)vctx;
    const int32_t pi = t / c->in_c;
    const int32_t ch = t % c->in_c;
    *x = ba_in[(size_t)c->pos[pi] * c->in_c + ch];
    *w = ba_w[((size_t)c->oc * 9 + c->tap[pi]) * c->in_c + ch];
}

static bool ba_direct_packed;

// The in-range taps of output `idx`, in the kernel's order (row by row).
static void ba_direct_taps_of(ba_direct *c, int32_t idx)
{
    const int32_t px = idx / c->out_c;
    const int32_t oy = px / c->out_hw;
    const int32_t ox = px % c->out_hw;
    c->oc = idx % c->out_c;
    c->n_pos = 0;
    for (int32_t ky = 0; ky < 3; ++ky)
    {
        for (int32_t kx = 0; kx < 3; ++kx)
        {
            const int32_t iy = oy - c->pad + ky * c->dil;
            const int32_t ix = ox - c->pad + kx * c->dil;
            if (iy >= 0 && iy < c->hw && ix >= 0 && ix < c->hw)
            {
                c->pos[c->n_pos] = iy * c->hw + ix;
                c->tap[c->n_pos] = ky * 3 + kx;
                ++c->n_pos;
            }
        }
    }
}

// The direct kernels skip padded taps: an output's reduction is its in-range taps only.
static int32_t ba_direct_taps(const void *vctx, int32_t idx)
{
    ba_direct c = *(const ba_direct *)vctx;
    ba_direct_taps_of(&c, idx);
    return c.n_pos * c.in_c;
}

static float16_t ba_direct_ref(const void *vctx, int32_t idx, int32_t block)
{
    ba_direct c = *(const ba_direct *)vctx;
    ba_direct_taps_of(&c, idx);
    const int32_t k = c.n_pos * c.in_c;
    if (ba_direct_packed)
    {
        return ba_emu_lane(&c, ba_direct_lane_tap, k, &ba_bias[c.oc], block);
    }
    return ba_emu_reduce(&c, ba_direct_red_tap, c.n_pos * ((c.in_c + 7) / 8), 1, &ba_bias[c.oc], k > block, block);
}

static void ba_direct_case(int32_t hw, int32_t pad, int32_t dil, int32_t in_c, int32_t out_c, bool packed)
{
    const int32_t out_hw = hw + 2 * pad - 2 * dil;
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {1, hw, hw, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 3, 3, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, out_hw, out_hw, out_c};
    cmsis_nn_conv_params_f16 p;
    char what[64];

    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.padding.h = pad;
    p.padding.w = pad;
    p.dilation.h = dil;
    p.dilation.w = dil;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
    TEST_ASSERT_TRUE(hw * hw * in_c <= BA_IN_MAX && ((out_c + 7) / 8) * 8 * 9 * in_c <= BA_W_MAX &&
                     out_hw * out_hw * out_c <= BA_OUT_MAX);
    ba_fill(ba_in, hw * hw * in_c, (uint32_t)in_c + 51u);
    ba_fill(ba_w, out_c * 9 * in_c, (uint32_t)in_c + 52u);
    ba_fill(ba_bias, out_c, (uint32_t)in_c + 53u);
    const float16_t *w = ba_w;
    if (packed)
    {
        ba_pack_nt_n(ba_w, ba_wp, out_c, 9 * in_c);
        w = ba_wp;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f16(&ctx,
                                       &p,
                                       &input_dims,
                                       ba_in,
                                       &filter_dims,
                                       w,
                                       &bias_dims,
                                       ba_bias,
                                       &output_dims,
                                       ba_out,
                                       ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f16_acc16(&ctx,
                                             &p,
                                             &input_dims,
                                             ba_in,
                                             &filter_dims,
                                             w,
                                             &bias_dims,
                                             ba_bias,
                                             &output_dims,
                                             ba_out16,
                                             ARM_NN_LAYOUT_NHWC));
    /* The float16-lane wrapper runs the same kernel as the entry. */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_f16_acc16(&ctx,
                                                     &p,
                                                     &input_dims,
                                                     ba_in,
                                                     &filter_dims,
                                                     w,
                                                     &bias_dims,
                                                     ba_bias,
                                                     &output_dims,
                                                     ba_out_wrap));
    TEST_ASSERT_EQUAL_MEMORY(ba_out16, ba_out_wrap, (size_t)(out_hw * out_hw * out_c) * sizeof(float16_t));
    ba_direct c;
    memset(&c, 0, sizeof(c));
    c.hw = hw;
    c.out_hw = out_hw;
    c.pad = pad;
    c.dil = dil;
    c.in_c = in_c;
    c.out_c = out_c;
    ba_direct_packed = packed;
    snprintf(what, sizeof(what), "direct%s c%ld p%ld d%ld", packed ? "-packed" : "", (long)in_c, (long)pad, (long)dil);
    ba_expect_taps(what, out_hw * out_hw * out_c, ba_direct_ref, &c, 9 * in_c, ba_direct_taps);
}

void ba_conv_direct_long_k(void)
{
    ba_begin();
    ba_direct_case(4, 1, 1, 8, 5, false);  /* K = 72: one vector step per tap; corners 32 taps, edges 48 */
    ba_direct_case(4, 1, 1, 40, 5, false); /* K = 360: blocks end inside a tap */
    ba_direct_case(4, 1, 1, 13, 5, true);  /* K = 117 */
    ba_direct_case(4, 1, 1, 40, 5, true);
    ba_end();
}

/* Padded outputs of a long patch whose own in-range taps are at most 32: the direct kernels skip the padded taps, so
 * those outputs keep the float16 result. */
void ba_conv_direct_edges(void)
{
    ba_begin();
    ba_direct_case(4, 2, 1, 8, 5, false);  /* K = 72; 1..9 in-range taps: 8, 16, 24, 32, 48, 72 */
    ba_direct_case(5, 1, 1, 9, 5, false);  /* K = 81; corners 36, edges 54 */
    ba_direct_case(3, 2, 1, 11, 5, false); /* K = 99; 11, 22, 33, 44, 66, 99 */
    ba_direct_case(5, 2, 2, 8, 5, false);  /* dilation 2: 4, 6 or 9 in-range taps, 32, 48 or 72 */
    ba_direct_case(4, 2, 1, 8, 5, true);
    ba_direct_case(3, 2, 1, 11, 13, true);
    ba_direct_case(5, 2, 2, 8, 5, true);
    ba_end();
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Depthwise: ch_mult == 1 direct kernel, ch_mult > 1 generic kernel, one-channel to-conv route                    */
/* ---------------------------------------------------------------------------------------------------------------- */

typedef struct
{
    int32_t hw;
    int32_t k;
    int32_t in_c;
    int32_t mult;
    bool zero_taps; /* the to-conv route reads the padded taps as zeros */
    int32_t oy;
    int32_t ox;
    int32_t oc;
    int32_t taps[64];
    int32_t n;
} ba_dw;

static void ba_dw_lane_tap(const void *vctx, int32_t t, float16_t *x, float16_t *w)
{
    const ba_dw *c = (const ba_dw *)vctx;
    const int32_t tap = c->taps[t];
    const int32_t iy = c->oy - c->k / 2 + tap / c->k;
    const int32_t ix = c->ox - c->k / 2 + tap % c->k;
    const bool in = iy >= 0 && iy < c->hw && ix >= 0 && ix < c->hw;
    const int32_t ic = c->oc / c->mult;
    *x = ba_from_bits(in ? ba_bits(ba_in[((size_t)iy * c->hw + ix) * c->in_c + ic]) : 0u);
    *w = ba_w[(size_t)tap * c->in_c * c->mult + c->oc];
}

// The taps of output `idx`: the in-range ones, or all of them on the to-conv route.
static void ba_dw_taps_of(ba_dw *c, int32_t idx)
{
    const int32_t out_c = c->in_c * c->mult;
    const int32_t px = idx / out_c;
    c->oy = px / c->hw;
    c->ox = px % c->hw;
    c->oc = idx % out_c;
    c->n = 0;
    for (int32_t tap = 0; tap < c->k * c->k; ++tap)
    {
        const int32_t iy = c->oy - c->k / 2 + tap / c->k;
        const int32_t ix = c->ox - c->k / 2 + tap % c->k;
        if (c->zero_taps || (iy >= 0 && iy < c->hw && ix >= 0 && ix < c->hw))
        {
            c->taps[c->n++] = tap;
        }
    }
}

static int32_t ba_dw_taps(const void *vctx, int32_t idx)
{
    ba_dw c = *(const ba_dw *)vctx;
    ba_dw_taps_of(&c, idx);
    return c.n;
}

static float16_t ba_dw_ref(const void *vctx, int32_t idx, int32_t block)
{
    ba_dw c = *(const ba_dw *)vctx;
    ba_dw_taps_of(&c, idx);
    return ba_emu_lane(&c, ba_dw_lane_tap, c.n, &ba_bias[c.oc], block);
}

static void ba_dw_case(int32_t hw, int32_t k, int32_t in_c, int32_t mult, bool with_ctx)
{
    const int32_t out_c = in_c * mult;
    const cmsis_nn_dims input_dims = {1, hw, hw, in_c};
    const cmsis_nn_dims filter_dims = {1, k, k, out_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, hw, hw, out_c};
    cmsis_nn_dw_conv_params_f16 p;
    char what[64];

    memset(&p, 0, sizeof(p));
    p.ch_mult = mult;
    p.stride.h = 1;
    p.stride.w = 1;
    p.padding.h = k / 2;
    p.padding.w = k / 2;
    p.dilation.h = 1;
    p.dilation.w = 1;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    const cmsis_nn_context ctx = {with_ctx ? ba_scratch : NULL, with_ctx ? (int32_t)sizeof(ba_scratch) : 0};
    TEST_ASSERT_TRUE(hw * hw * in_c <= BA_IN_MAX && k * k * out_c <= BA_W_MAX && hw * hw * out_c <= BA_OUT_MAX);
    ba_fill(ba_in, hw * hw * in_c, (uint32_t)(k * 100 + in_c) + 61u);
    ba_fill(ba_w, k * k * out_c, (uint32_t)(k * 100 + in_c) + 62u);
    ba_fill(ba_bias, out_c, (uint32_t)(k * 100 + in_c) + 63u);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_f16(&ctx,
                                             &p,
                                             &input_dims,
                                             ba_in,
                                             &filter_dims,
                                             ba_w,
                                             &bias_dims,
                                             ba_bias,
                                             &output_dims,
                                             ba_out,
                                             ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_f16_acc16(&ctx,
                                                   &p,
                                                   &input_dims,
                                                   ba_in,
                                                   &filter_dims,
                                                   ba_w,
                                                   &bias_dims,
                                                   ba_bias,
                                                   &output_dims,
                                                   ba_out16,
                                                   ARM_NN_LAYOUT_NHWC));
    /* The float16-lane wrapper runs the same kernel as the entry. */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_f16_acc16(&ctx,
                                                           &p,
                                                           &input_dims,
                                                           ba_in,
                                                           &filter_dims,
                                                           ba_w,
                                                           &bias_dims,
                                                           ba_bias,
                                                           &output_dims,
                                                           ba_out_wrap));
    TEST_ASSERT_EQUAL_MEMORY(ba_out16, ba_out_wrap, (size_t)(hw * hw * out_c) * sizeof(float16_t));
    ba_dw c;
    memset(&c, 0, sizeof(c));
    c.hw = hw;
    c.k = k;
    c.in_c = in_c;
    c.mult = mult;
    c.zero_taps = BA_MVE && with_ctx && in_c == 1 && out_c >= 8;
    snprintf(what, sizeof(what), "dw k%ld c%ld x%ld", (long)k, (long)in_c, (long)mult);
    if (!BA_MVE && mult > 1)
    {
        /* The generic kernel's scalar legs keep their float16 accumulator. */
        ba_expect_same(what, hw * hw * out_c);
        return;
    }
    ba_expect_taps(what, hw * hw * out_c, ba_dw_ref, &c, k * k, ba_dw_taps);
}

void ba_depthwise_long_k(void)
{
    ba_begin();
    ba_dw_case(9, 7, 20, 1, false); /* 49 taps inside, 16..42 at the edges; channel blocks 16 + 4 */
    ba_dw_case(8, 7, 8, 1, false);
    ba_dw_case(7, 7, 3, 2, false);  /* generic kernel */
    ba_dw_case(8, 7, 1, 9, true);   /* to-conv route: 49 taps with the padding read as zeros */
    ba_dw_case(8, 5, 20, 1, false); /* 25 taps: nothing folds */
    ba_dw_case(7, 5, 3, 2, false);
    ba_dw_case(9, 7, 13, 1, false); /* one two-vector channel block with a predicated tail */
    ba_dw_case(9, 7, 29, 1, false);
    ba_end();
}

/* 8x8 kernels: 64 taps inside, exactly 32 on the edges where 4 of the 8 rows (or columns) are in range. */
void ba_depthwise_edges(void)
{
    ba_begin();
    ba_dw_case(9, 8, 12, 1, false);
    ba_dw_case(9, 8, 3, 2, false);
    ba_dw_case(9, 8, 1, 9, true); /* to-conv: the padded taps count, so every output has 64 */
    ba_end();
}
