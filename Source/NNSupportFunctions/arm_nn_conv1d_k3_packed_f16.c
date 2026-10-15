/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k3_packed_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 3 for packed f16 weights
 *
 * $Date:        29 Apr 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

__STATIC_FORCEINLINE void arm_nn_conv1d_k3_packed_f16_body(const float16_t *__RESTRICT x_nhwc,
                                                           int32_t in_c,
                                                           int32_t in_w,
                                                           const float16_t *__RESTRICT kernel_packed,
                                                           const float16_t *__RESTRICT b,
                                                           float16_t *__RESTRICT out,
                                                           int32_t out_c,
                                                           int32_t out_w,
                                                           const int32_t block)
{
    (void)in_w;
    (void)block;

    const int32_t block_cols = 8;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* Each lane takes 3 taps per input channel; with more than `block` taps a lane's float16 partial covers at most
     * block / 3 input channels before it is widened into per-lane float32 accumulators (#586). */
    const bool fold = 3 * in_c > block;
    const int32_t span = fold ? block / 3 : in_c;
    int32_t ow = 0;
    for (; ow + 1 < out_w; ow += 2)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        float16_t *y0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *y1 = out + (size_t)(ow + 1) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;

            float32x4_t vacc0_sum_even = vdupq_n_f32(0.0f);
            float32x4_t vacc0_sum_odd = vdupq_n_f32(0.0f);
            float32x4_t vacc1_sum_even = vdupq_n_f32(0.0f);
            float32x4_t vacc1_sum_odd = vdupq_n_f32(0.0f);
            int32_t ic = 0;
            for (;;)
            {
                const int32_t end = (in_c - ic > span) ? ic + span : in_c;
                for (; ic < end; ++ic)
                {
                    const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                    const float16x8_t vw0 = vld1q(w_ic + 0U * in_c * block_cols);
                    const float16x8_t vw1 = vld1q(w_ic + 1U * in_c * block_cols);
                    const float16x8_t vw2 = vld1q(w_ic + 2U * in_c * block_cols);

                    vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                    vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                    vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                    vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                    vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                    vacc1 = vfmaq(vacc1, vw2, x3[ic]);
                }
                if (!fold)
                {
                    break;
                }
                arm_nn_f16_fold_lanes_f32(&vacc0_sum_even, &vacc0_sum_odd, vacc0, end == span);
                arm_nn_f16_fold_lanes_f32(&vacc1_sum_even, &vacc1_sum_odd, vacc1, end == span);
                if (ic == in_c)
                {
                    vacc0 = arm_nn_f16_narrow_lanes_f32(vacc0_sum_even, vacc0_sum_odd);
                    vacc1 = arm_nn_f16_narrow_lanes_f32(vacc1_sum_even, vacc1_sum_odd);
                    break;
                }
                vacc0 = vdupq_n_f16((float16_t)0.0f);
                vacc1 = vdupq_n_f16((float16_t)0.0f);
            }

            vst1q(y0 + oc, vacc0);
            vst1q(y1 + oc, vacc1);
        }

        if (oc < out_c)
        {
            const int32_t valid_cols = out_c - oc;
            const mve_pred16_t p = vctp16q((uint32_t)valid_cols);
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;

            float32x4_t vacc0_sum_even = vdupq_n_f32(0.0f);
            float32x4_t vacc0_sum_odd = vdupq_n_f32(0.0f);
            float32x4_t vacc1_sum_even = vdupq_n_f32(0.0f);
            float32x4_t vacc1_sum_odd = vdupq_n_f32(0.0f);
            int32_t ic = 0;
            for (;;)
            {
                const int32_t end = (in_c - ic > span) ? ic + span : in_c;
                for (; ic < end; ++ic)
                {
                    const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                    const float16x8_t vw0 = vld1q_z(w_ic + 0U * in_c * block_cols, p);
                    const float16x8_t vw1 = vld1q_z(w_ic + 1U * in_c * block_cols, p);
                    const float16x8_t vw2 = vld1q_z(w_ic + 2U * in_c * block_cols, p);

                    vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                    vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                    vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                    vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                    vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                    vacc1 = vfmaq(vacc1, vw2, x3[ic]);
                }
                if (!fold)
                {
                    break;
                }
                arm_nn_f16_fold_lanes_f32(&vacc0_sum_even, &vacc0_sum_odd, vacc0, end == span);
                arm_nn_f16_fold_lanes_f32(&vacc1_sum_even, &vacc1_sum_odd, vacc1, end == span);
                if (ic == in_c)
                {
                    vacc0 = arm_nn_f16_narrow_lanes_f32(vacc0_sum_even, vacc0_sum_odd);
                    vacc1 = arm_nn_f16_narrow_lanes_f32(vacc1_sum_even, vacc1_sum_odd);
                    break;
                }
                vacc0 = vdupq_n_f16((float16_t)0.0f);
                vacc1 = vdupq_n_f16((float16_t)0.0f);
            }

            vst1q_p(y0 + oc, vacc0, p);
            vst1q_p(y1 + oc, vacc1, p);
        }
    }

    if (ow < out_w)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);

            float32x4_t vacc_sum_even = vdupq_n_f32(0.0f);
            float32x4_t vacc_sum_odd = vdupq_n_f32(0.0f);
            int32_t ic = 0;
            for (;;)
            {
                const int32_t end = (in_c - ic > span) ? ic + span : in_c;
                for (; ic < end; ++ic)
                {
                    const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                    vacc = vfmaq(vacc, vld1q(w_ic + 0U * in_c * block_cols), x0[ic]);
                    vacc = vfmaq(vacc, vld1q(w_ic + 1U * in_c * block_cols), x1[ic]);
                    vacc = vfmaq(vacc, vld1q(w_ic + 2U * in_c * block_cols), x2[ic]);
                }
                if (!fold)
                {
                    break;
                }
                arm_nn_f16_fold_lanes_f32(&vacc_sum_even, &vacc_sum_odd, vacc, end == span);
                if (ic == in_c)
                {
                    vacc = arm_nn_f16_narrow_lanes_f32(vacc_sum_even, vacc_sum_odd);
                    break;
                }
                vacc = vdupq_n_f16((float16_t)0.0f);
            }

            vst1q(y + oc, vacc);
        }

        if (oc < out_c)
        {
            const int32_t valid_cols = out_c - oc;
            const mve_pred16_t p = vctp16q((uint32_t)valid_cols);
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);

            float32x4_t vacc_sum_even = vdupq_n_f32(0.0f);
            float32x4_t vacc_sum_odd = vdupq_n_f32(0.0f);
            int32_t ic = 0;
            for (;;)
            {
                const int32_t end = (in_c - ic > span) ? ic + span : in_c;
                for (; ic < end; ++ic)
                {
                    const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                    vacc = vfmaq(vacc, vld1q_z(w_ic + 0U * in_c * block_cols, p), x0[ic]);
                    vacc = vfmaq(vacc, vld1q_z(w_ic + 1U * in_c * block_cols, p), x1[ic]);
                    vacc = vfmaq(vacc, vld1q_z(w_ic + 2U * in_c * block_cols, p), x2[ic]);
                }
                if (!fold)
                {
                    break;
                }
                arm_nn_f16_fold_lanes_f32(&vacc_sum_even, &vacc_sum_odd, vacc, end == span);
                if (ic == in_c)
                {
                    vacc = arm_nn_f16_narrow_lanes_f32(vacc_sum_even, vacc_sum_odd);
                    break;
                }
                vacc = vdupq_n_f16((float16_t)0.0f);
            }

            vst1q_p(y + oc, vacc, p);
        }
    }
    #else
    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const int32_t lane = oc % block_cols;
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            /* Accumulate in float32 and round to f16 once at the store (#449, #465). */
            float32_t acc = b ? (float32_t)b[oc] : 0.0f;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                acc += (float32_t)x0[ic] * (float32_t)w_ic[(size_t)0 * in_c * block_cols + lane];
                acc += (float32_t)x1[ic] * (float32_t)w_ic[(size_t)1 * in_c * block_cols + lane];
                acc += (float32_t)x2[ic] * (float32_t)w_ic[(size_t)2 * in_c * block_cols + lane];
            }

            y[oc] = (float16_t)acc;
        }
    }
    #endif
}

void arm_nn_conv1d_k3_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w)
{
    arm_nn_conv1d_k3_packed_f16_body(x_nhwc, in_c, in_w, kernel_packed, b, out, out_c, out_w, ARM_NN_F16_ACC_BLOCK);
}

void arm_nn_conv1d_k3_packed_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t in_c,
                                       int32_t in_w,
                                       const float16_t *__RESTRICT kernel_packed,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t out_c,
                                       int32_t out_w)
{
    arm_nn_conv1d_k3_packed_f16_body(
        x_nhwc, in_c, in_w, kernel_packed, b, out, out_c, out_w, ARM_NN_F16_ACC_BLOCK_NONE);
}

#endif /* ARM_NN_ENABLE_F16 */
