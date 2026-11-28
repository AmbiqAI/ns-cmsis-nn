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

    /* These monotonic keys map both signed zeros to the midpoint. Integer comparisons
     * retain subnormal ordering with MVE and NaN handling under finite-math-only. */
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
static inline uint32x4_t order_key(uint32x4_t bits)
{
    const uint32x4_t midpoint = vdupq_n_u32(0x80000000u);
    return vpselq(vsubq(vdupq_n_u32(0), bits), veorq(bits, midpoint), vcmpcsq_n_u32(bits, 0x80000000u));
}
    #else
static inline uint32_t order_key(uint32_t bits) { return bits >= 0x80000000u ? 0u - bits : bits ^ 0x80000000u; }
    #endif

__STATIC_FORCEINLINE void
compare_span(const float32_t *a, const float32_t *b, bool *out, int32_t count, bool scalar_a, bool scalar_b)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t i = 0; i < count; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(count - i));
        const uint32x4_t x = scalar_a ? vdupq_n_u32(load_bits(a)) : arm_nn_mve_load_word_bits(a + i, p);
        const uint32x4_t y = scalar_b ? vdupq_n_u32(load_bits(b)) : arm_nn_mve_load_word_bits(b + i, p);
        const uint32x4_t magnitude = vdupq_n_u32(0x7fffffffu);
        const uint32x4_t largest = vmaxq(vandq(x, magnitude), vandq(y, magnitude));
        const mve_pred16_t ordered = vcmpcsq(vdupq_n_u32(0x7f800000u), largest);
        const mve_pred16_t greater = vcmphiq(order_key(x), order_key(y));
        vstrbq_p_u32((uint8_t *)out + i, vpselq(vdupq_n_u32(1), vdupq_n_u32(0), (mve_pred16_t)(ordered & greater)), p);
    }
    #else
    for (int32_t i = 0; i < count; ++i)
    {
        const uint32_t x = load_bits(a + (scalar_a ? 0 : i));
        const uint32_t y = load_bits(b + (scalar_b ? 0 : i));
        out[i] = (x & 0x7fffffffu) <= 0x7f800000u && (y & 0x7fffffffu) <= 0x7f800000u && order_key(x) > order_key(y);
    }
    #endif
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
