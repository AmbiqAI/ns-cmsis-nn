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
                               void *dst)
{
    const float16_t *tl_data = top_left, *bl_data = bottom_left, *tr_data = top_right, *br_data = bottom_right;
    float16_t *out = dst;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t c = 0; c < channels; c += 8)
    {
        // Cap the explicit load/store predicate at one vector. GCC 15 otherwise
        // emits DLSTP.16 across the widened FP32 arithmetic and drops a live tail lane.
        const uint32_t lanes = (uint32_t)(channels - c);
        const mve_pred16_t p = vctp16q(lanes < 8U ? lanes : 8U);
        const float16x8_t tl = vld1q_z(tl_data + c, p);
        const float16x8_t bl = vld1q_z(bl_data + c, p);
        const float16x8_t tr = vld1q_z(tr_data + c, p);
        const float16x8_t br = vld1q_z(br_data + c, p);
        float32x4_t even = vmulq_n_f32(arm_nn_vcvtbq_f32_f16(tl), w_tl);
        float32x4_t odd = vmulq_n_f32(arm_nn_vcvttq_f32_f16(tl), w_tl);
        even = vfmaq_n_f32(even, arm_nn_vcvtbq_f32_f16(bl), w_bl);
        odd = vfmaq_n_f32(odd, arm_nn_vcvttq_f32_f16(bl), w_bl);
        even = vfmaq_n_f32(even, arm_nn_vcvtbq_f32_f16(tr), w_tr);
        odd = vfmaq_n_f32(odd, arm_nn_vcvttq_f32_f16(tr), w_tr);
        even = vfmaq_n_f32(even, arm_nn_vcvtbq_f32_f16(br), w_br);
        odd = vfmaq_n_f32(odd, arm_nn_vcvttq_f32_f16(br), w_br);
        float16x8_t result = arm_nn_vcvtbq_f16_f32(vdupq_n_f16(0), even);
        vstrhq_p(out + c, arm_nn_vcvttq_f16_f32(result, odd), p);
    }
    #else
    for (int32_t c = 0; c < channels; ++c)
    {
        out[c] = (float16_t)((float)tl_data[c] * w_tl + (float)bl_data[c] * w_bl + (float)tr_data[c] * w_tr +
                             (float)br_data[c] * w_br);
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
