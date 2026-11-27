/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32
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
                               void *dst)
{
    const float32_t *tl_data = top_left, *bl_data = bottom_left, *tr_data = top_right, *br_data = bottom_right;
    float32_t *out = dst;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t c = 0; c < channels; c += 4)
    {
        const mve_pred16_t p = vctp32q(channels - c);
        const float32x4_t tl = vld1q_z(tl_data + c, p);
        float32x4_t acc = vmulq_n_f32(tl, w_tl);
        acc = vfmaq_n_f32(acc, vld1q_z(bl_data + c, p), w_bl);
        acc = vfmaq_n_f32(acc, vld1q_z(tr_data + c, p), w_tr);
        acc = vfmaq_n_f32(acc, vld1q_z(br_data + c, p), w_br);
        vstrwq_p(out + c, acc, p);
    }
    #else
    for (int32_t c = 0; c < channels; ++c)
    {
        out[c] = (float32_t)((float)tl_data[c] * w_tl + (float)bl_data[c] * w_bl + (float)tr_data[c] * w_tr +
                             (float)br_data[c] * w_br);
    }
    #endif
}

arm_cmsis_nn_status arm_resize_bilinear_f32(const cmsis_nn_resize_params *resize_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float32_t *input_data,
                                            const cmsis_nn_dims *output_dims,
                                            float32_t *output_data)
{
    return arm_nn_bilinear_resize(
        resize_params, input_dims, input_data, output_dims, output_data, sizeof(float32_t), interpolate);
}
#endif
