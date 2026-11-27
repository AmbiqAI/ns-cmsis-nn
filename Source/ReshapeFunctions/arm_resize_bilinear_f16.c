/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16
    #include "Internal/arm_nn_resize_bilinear_flt.h"

static inline void interpolate(const void *top_left,
                               const void *bottom_left,
                               const void *top_right,
                               const void *bottom_right,
                               float w_tl,
                               float w_bl,
                               float w_tr,
                               float w_br,
                               int32_t channels,
                               void *dst,
                               bool quarter_grid)
{
    const float16_t *tl_data = top_left, *bl_data = bottom_left, *tr_data = top_right, *br_data = bottom_right;
    float16_t *out = dst;
    const _Float16 tl_weight = (_Float16)w_tl;
    const _Float16 bl_weight = (_Float16)w_bl;
    const _Float16 tr_weight = (_Float16)w_tr;
    const _Float16 br_weight = (_Float16)w_br;
    // Rounded weights can sum to more than one. Keep finite samples in binary16 range.
    const _Float16 limit = (_Float16)65504.0f;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    // With nonnegative weights MVE's fixed multiply/FMA chain is monotone in each
    // sample. For the 16 quarter-fraction pairs even four signed finite extrema
    // stay in range (covered by the extreme-grid test), so these grids need no clamp.
    const float16x8_t lower = vdupq_n_f16(-limit);
    const float16x8_t upper = vdupq_n_f16(limit);
    for (int32_t c = 0; c < channels; c += 8)
    {
        const mve_pred16_t p = vctp16q(channels - c);
        float16x8_t acc = vmulq_n_f16(vld1q_z(tl_data + c, p), tl_weight);
        acc = vfmaq_n_f16(acc, vld1q_z(bl_data + c, p), bl_weight);
        acc = vfmaq_n_f16(acc, vld1q_z(tr_data + c, p), tr_weight);
        acc = vfmaq_n_f16(acc, vld1q_z(br_data + c, p), br_weight);
        vstrhq_p(out + c, (quarter_grid ? acc : vminnmq(vmaxnmq(acc, lower), upper)), p);
    }
    #else
    (void)quarter_grid;
    for (int32_t c = 0; c < channels; ++c)
    {
        _Float16 acc = (_Float16)tl_data[c] * tl_weight;
        acc += (_Float16)bl_data[c] * bl_weight;
        acc += (_Float16)tr_data[c] * tr_weight;
        acc += (_Float16)br_data[c] * br_weight;
        out[c] = arm_nn_clamp_f16h(acc, limit, -limit);
    }
    #endif
}

arm_cmsis_nn_status arm_resize_bilinear_f16(const cmsis_nn_resize_params *resize_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float16_t *input_data,
                                            const cmsis_nn_dims *output_dims,
                                            float16_t *output_data)
{
    return arm_nn_bilinear_resize(
        resize_params, input_dims, input_data, output_dims, output_data, sizeof(float16_t), interpolate);
}
#endif
