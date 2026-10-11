/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#define __STATIC_FORCEINLINE static constexpr inline
#define ARM_MATH_MVEI 1
#include "mve_shim.h"

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"
#define ARM_NNSUPPORTFUNCTIONS_H /* Use the constexpr helpers below instead of the runtime ones. */

/* Compile the production MVE row-broadcast blocks of arm_mul_s8() and arm_add_s8() as constant expressions. Only
 * the helpers they call are replaced, with the same arithmetic; their pointer math is not adapted. */
#define ARM_NN_MAX(A, B) ((A) > (B) ? (A) : (B))
#define ARM_NN_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LEFT_SHIFT(_shift) (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)
constexpr int32_t arm_nn_is_row_broadcast(const cmsis_nn_dims *input1_dims,
                                          const cmsis_nn_dims *input2_dims,
                                          const cmsis_nn_dims *output_dims)
{
    return input1_dims->c == input2_dims->c && input1_dims->c > 1 && (input1_dims->w == 1) != (input2_dims->w == 1) &&
        output_dims->w > 1;
}
constexpr int32x4_t arm_divide_by_nonzero_power_of_two_mve(const int32x4_t dividend, const int32x4_t neg_exp)
{
    return vrshlq_s32(vqaddq_s32(dividend, vshrq_n_s32(dividend, 31)), neg_exp);
}
constexpr int32x4_t arm_requantize_mve(const int32x4_t val, const int32_t multiplier, const int32_t shift)
{
    const int32x4_t neg_exp = vdupq_n_s32(-RIGHT_SHIFT(shift));
    const int32x4_t fixup = vshrq_n_s32(vandq_s32(vqrdmulhq_n_s32(vshlq_r_s32(val, LEFT_SHIFT(shift)), multiplier),
                                                  neg_exp),
                                        31);
    return vrshlq_s32(
        vqaddq_s32(vqrdmulhq_n_s32(vshlq_r_s32(val, LEFT_SHIFT(shift)), multiplier), fixup), neg_exp);
}
constexpr int32_t probe_requantize(int32_t val, int32_t mult, int32_t shift)
{
    const int32_t high = shim_qrdmulh(shim_shl(val, LEFT_SHIFT(shift)), mult);
    const int32_t exp = RIGHT_SHIFT(shift);
    const int32_t mask = (int32_t)(((int64_t)1 << exp) - 1);
    const int32_t threshold = (mask >> 1) + (high < 0 ? 1 : 0);
    return (high >> exp) + ((high & mask) > threshold ? 1 : 0);
}
#include "arm_add_s8.c"
#include "arm_mul_s8.c"

/* Full operand [N, H, W, C] against a row operand [N, H, 1, C], each array ending at its last element, in either
 * argument order, against a scalar reference. */
template <bool Add, int N, int H, int W, int C> constexpr bool row_broadcast(bool swap, int32_t out_shift)
{
    int8_t full[N * H * W * C] = {};
    int8_t row[N * H * C] = {};
    int8_t out[N * H * W * C] = {};
    for (int i = 0; i < N * H * W * C; ++i)
        full[i] = (int8_t)(i * 37 - 90);
    for (int i = 0; i < N * H * C; ++i)
        row[i] = (int8_t)(i * 53 + 11);
    const cmsis_nn_dims full_dims = {N, H, W, C};
    const cmsis_nn_dims row_dims = {N, H, 1, C};
    const cmsis_nn_dims out_dims = {N, H, W, C};
    const int8_t *in1 = swap ? row : full;
    const int8_t *in2 = swap ? full : row;
    const cmsis_nn_dims *dims1 = swap ? &row_dims : &full_dims;
    const cmsis_nn_dims *dims2 = swap ? &full_dims : &row_dims;
    const int32_t off1 = 3, off2 = -5, mult1 = 1623821475, shift1 = swap ? -1 : 0, mult2 = 1310720000, shift2 = -2,
                  left_shift = 20, out_mult = 1173388748, out_offset = -2;
    if (Add)
        arm_add_s8_row_broadcast_rows(in1, dims1, in2, dims2, off1, mult1, shift1, off2, mult2, shift2, left_shift,
                                      out, &out_dims, out_offset, out_mult, out_shift, -128, 127);
    else
        arm_mul_s8_row_broadcast_rows(in1, dims1, in2, dims2, off1, off2, out, &out_dims, out_offset, out_mult,
                                      out_shift, -128, 127);
    for (int i = 0; i < N * H * W * C; ++i)
    {
        const int32_t a = in1 == full ? full[i] : row[(i / (W * C)) * C + i % C];
        const int32_t b = in2 == full ? full[i] : row[(i / (W * C)) * C + i % C];
        int32_t r;
        if (Add)
            r = probe_requantize(probe_requantize((a + off1) << left_shift, mult1, shift1) +
                                     probe_requantize((b + off2) << left_shift, mult2, shift2),
                                 out_mult,
                                 out_shift);
        else
            r = probe_requantize((a + off1) * (b + off2), out_mult, out_shift);
        r = ARM_NN_MIN(ARM_NN_MAX(r + out_offset, -128), 127);
        if (out[i] != r)
            return false;
    }
    return true;
}

static_assert(row_broadcast<false, 1, 2, 3, 12>(false, -7) && row_broadcast<false, 2, 1, 2, 9>(true, 1),
              "mul row-broadcast rows stay within their operands");
static_assert(row_broadcast<true, 1, 2, 3, 12>(false, -20) && row_broadcast<true, 2, 1, 2, 6>(true, -19),
              "add row-broadcast rows stay within their operands");
