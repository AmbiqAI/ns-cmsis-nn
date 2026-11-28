/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32
static inline uint32_t load_bits(const float32_t *input)
{
    uint32_t bits;
    arm_nn_copy_word(&bits, input);
    return bits;
}

/* Map signed zeros to the same key and preserve binary32 order for non-NaNs. */
static inline uint32_t order_key(uint32_t bits) { return bits >= 0x80000000u ? 0u - bits : bits ^ 0x80000000u; }

static __attribute__((noinline)) void
compare_storage(const float32_t *a, const float32_t *b, bool *out, int32_t count, int32_t step_a, int32_t step_b)
{
    for (int32_t i = 0; i < count; ++i)
    {
        const uint32_t x = load_bits(a + i * step_a);
        const uint32_t y = load_bits(b + i * step_b);
        out[i] = (x & 0x7fffffffu) <= 0x7f800000u && (y & 0x7fffffffu) <= 0x7f800000u && order_key(x) > order_key(y);
    }
}

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_FORCEINLINE uint32x4_t compare_four(const uint32x4_t x, const uint32x4_t y, bool *out, mve_pred16_t p)
{
    const mve_pred16_t gt = vcmpgtq_f32(vreinterpretq_f32_u32(x), vreinterpretq_f32_u32(y));
    vstrbq_p_u32((uint8_t *)out, vpselq(vdupq_n_u32(1), vdupq_n_u32(0), gt), p);
    return vmaxq(vshlq_n_u32(x, 1), vshlq_n_u32(y, 1));
}
    #elif defined(__ARM_FP) && (__ARM_FP & 4) && defined(__ARM_ARCH_7EM__) && (defined(__GNUC__) || defined(__clang__))
        #define ARM_NN_COMPARE_SCALAR_FPU 1
/* An equal VCMP result can hide differing subnormals when FPSCR.FZ is set.
 * Replay only that case using the original storage bits. A comparison with
 * either NaN reports unordered (both GT and EQ clear). */
static __attribute__((noinline)) bool compare_tie(float32_t a, float32_t b)
{
    return order_key(load_bits(&a)) > order_key(load_bits(&b));
}

static inline bool scalar_greater(float32_t a, float32_t b)
{
    /* Comparing b to a puts ordered b < a in N and equality in Z.
     * Read those bits directly to keep the same ordered test under fast-math. */
    uint32_t flags;
    __asm("vcmp.f32 %2, %1\n\tvmrs %0, fpscr" : "=r"(flags) : "t"(a), "t"(b));
    if (__builtin_expect((flags & (1u << 30)) != 0, false))
    {
        return compare_tie(a, b);
    }
    return (flags >> 31) != 0;
}

    #endif
__STATIC_FORCEINLINE void
compare_span(const float32_t *a, const float32_t *b, bool *restrict out, int32_t count, bool scalar_a, bool scalar_b)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    uint32x4_t largest = vdupq_n_u32(0);
    uint32x4_t smallest = vdupq_n_u32(UINT32_MAX);
    const uint32x4_t ax = scalar_a ? vdupq_n_u32(load_bits(a)) : vdupq_n_u32(0);
    const uint32x4_t bx = scalar_b ? vdupq_n_u32(load_bits(b)) : vdupq_n_u32(0);
    int32_t i = 0;
    for (; i <= count - 4; i += 4)
    {
        const uint32x4_t x = scalar_a ? ax : arm_nn_mve_load_word_bits(a + i, 0xffff);
        const uint32x4_t y = scalar_b ? bx : arm_nn_mve_load_word_bits(b + i, 0xffff);
        const uint32x4_t magnitude = compare_four(x, y, out + i, 0xffff);
        largest = vmaxq(largest, magnitude);
        smallest = vminq(smallest, vsubq_n_u32(magnitude, 1));
    }
    if (i < count)
    {
        const mve_pred16_t p = vctp32q(count - i);
        const uint32x4_t x = scalar_a ? ax : arm_nn_mve_load_word_bits(a + i, p);
        const uint32x4_t y = scalar_b ? bx : arm_nn_mve_load_word_bits(b + i, p);
        const uint32x4_t magnitude = compare_four(x, y, out + i, p);
        largest = vmaxq_m(largest, largest, magnitude, p);
        smallest = vminq_m(smallest, smallest, vsubq_n_u32(magnitude, 1), p);
    }
    // Doubled magnitudes clear the sign. Above infinity the native fast-math
    // compare can be unordered. Flushing to zero can change the order only
    // when both magnitudes are below the first normal. Subtracting one maps a
    // pair of exact zeros to UINT32_MAX so sparse zero runs stay on the fast path.
    if (vminvq(UINT32_MAX, smallest) >= 0x00ffffffu && vmaxvq(0, largest) <= 0xff000000u)
    {
        return;
    }
    #elif defined(ARM_NN_COMPARE_SCALAR_FPU)
    /* Store four consecutive one-byte results together, including when the
     * public mask starts at a byte-aligned offset. */
    int32_t i = 0;
    for (; i <= count - 4; i += 4)
    {
        bool group[4];
        for (int lane = 0; lane < 4; ++lane)
        {
            const float32_t x = a[scalar_a ? 0 : i + lane], y = b[scalar_b ? 0 : i + lane];
            group[lane] = scalar_greater(x, y);
        }
        __builtin_memcpy(out + i, group, sizeof(group));
    }
    for (; i < count; ++i)
    {
        const float32_t x = a[scalar_a ? 0 : i], y = b[scalar_b ? 0 : i];
        out[i] = scalar_greater(x, y);
    }
    return;
    #endif

    compare_storage(a, b, out, count, scalar_a ? 0 : 1, scalar_b ? 0 : 1);
}

/* The broadcast walker can visit a span from several geometry branches. Keep one
 * vector loop per operand layout, with the scalar choice constant in each. */
static __attribute__((noinline)) void compare_full(const float32_t *a, const float32_t *b, bool *out, int32_t count)
{
    compare_span(a, b, out, count, false, false);
}

static __attribute__((noinline)) void compare_scalar_a(const float32_t *a, const float32_t *b, bool *out, int32_t count)
{
    compare_span(a, b, out, count, true, false);
}

static __attribute__((noinline)) void compare_scalar_b(const float32_t *b, const float32_t *a, bool *out, int32_t count)
{
    compare_span(a, b, out, count, false, true);
}

arm_cmsis_nn_status arm_greater_f32(const float32_t *input_1,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float32_t *input_2,
                                    const cmsis_nn_dims *input_2_dims,
                                    bool *output,
                                    const cmsis_nn_dims *output_dims)
{
    if (!input_1 || !input_2 || !output || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims) ||
        !arm_nn_broadcast_bytes_fit(output_dims, sizeof(float32_t)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    ARM_NN_BROADCAST_WALK_NHWC(float32_t,
                               bool,
                               input_1,
                               input_1_dims,
                               input_2,
                               input_2_dims,
                               output,
                               output_dims,
                               compare_full,
                               compare_scalar_a,
                               compare_scalar_b);
    return ARM_CMSIS_NN_SUCCESS;
}
#endif
