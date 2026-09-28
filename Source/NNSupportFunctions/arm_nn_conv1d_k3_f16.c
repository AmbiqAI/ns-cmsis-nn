/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k3_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 3 for f16
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV1D_K3_NHWC_F16_OC4_BLOCK (4)
    #endif

__STATIC_FORCEINLINE void arm_nn_conv1d_k3_nhwc_f16_body(const float16_t *__RESTRICT x_nhwc,
                                                         int32_t in_c,
                                                         int32_t in_w,
                                                         const float16_t *__RESTRICT kernel,
                                                         const float16_t *__RESTRICT b,
                                                         float16_t *__RESTRICT out,
                                                         int32_t out_c,
                                                         int32_t out_w,
                                                         const int32_t block)
{
    (void)in_w;
    (void)block;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* Each lane takes 3 taps per input-channel vector. With more than `block` taps per output, a lane's float16
     * partial covers at most block / 3 channel vectors; each block's lanes are folded into float32 pair
     * accumulators (arm_nn_f16_fold_pairs_f32), summed once (arm_nn_f16_pairs_sum_f32), the bias is added in float32
     * and the total rounds to float16 once (#586). */
    const bool fold = 3 * in_c > block;
    const int32_t span = fold ? (block / 3) * 8 : in_c;
    #endif

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t oc = 0;

        for (; oc + ARM_NN_CONV1D_K3_NHWC_F16_OC4_BLOCK <= out_c; oc += ARM_NN_CONV1D_K3_NHWC_F16_OC4_BLOCK)
        {
            const float16_t *w_base0 = kernel + (size_t)(oc + 0) * 3U * (size_t)in_c;
            const float16_t *w_base1 = kernel + (size_t)(oc + 1) * 3U * (size_t)in_c;
            const float16_t *w_base2 = kernel + (size_t)(oc + 2) * 3U * (size_t)in_c;
            const float16_t *w_base3 = kernel + (size_t)(oc + 3) * 3U * (size_t)in_c;
            float16x8_t vacc0 = vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc2 = vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc3 = vdupq_n_f16((float16_t)0.0f);
            float32x4_t vacc0_pairs = vdupq_n_f32(0.0f);
            float32x4_t vacc1_pairs = vdupq_n_f32(0.0f);
            float32x4_t vacc2_pairs = vdupq_n_f32(0.0f);
            float32x4_t vacc3_pairs = vdupq_n_f32(0.0f);

            int32_t ic0 = 0;
            do
            {
                const int32_t ic_end = (in_c - ic0 > span) ? ic0 + span : in_c;
                for (int32_t ic = ic0; ic < ic_end; ic += 8)
                {
                    const mve_pred16_t p = vctp16q((uint32_t)(ic_end - ic));
                    const float16x8_t vx0 = vld1q_z(x0 + ic, p);
                    const float16x8_t vx1 = vld1q_z(x1 + ic, p);
                    const float16x8_t vx2 = vld1q_z(x2 + ic, p);

                    vacc0 = vfmaq(vacc0, vx0, vld1q_z(w_base0 + 0 * in_c + ic, p));
                    vacc0 = vfmaq(vacc0, vx1, vld1q_z(w_base0 + 1 * in_c + ic, p));
                    vacc0 = vfmaq(vacc0, vx2, vld1q_z(w_base0 + 2 * in_c + ic, p));

                    vacc1 = vfmaq(vacc1, vx0, vld1q_z(w_base1 + 0 * in_c + ic, p));
                    vacc1 = vfmaq(vacc1, vx1, vld1q_z(w_base1 + 1 * in_c + ic, p));
                    vacc1 = vfmaq(vacc1, vx2, vld1q_z(w_base1 + 2 * in_c + ic, p));

                    vacc2 = vfmaq(vacc2, vx0, vld1q_z(w_base2 + 0 * in_c + ic, p));
                    vacc2 = vfmaq(vacc2, vx1, vld1q_z(w_base2 + 1 * in_c + ic, p));
                    vacc2 = vfmaq(vacc2, vx2, vld1q_z(w_base2 + 2 * in_c + ic, p));

                    vacc3 = vfmaq(vacc3, vx0, vld1q_z(w_base3 + 0 * in_c + ic, p));
                    vacc3 = vfmaq(vacc3, vx1, vld1q_z(w_base3 + 1 * in_c + ic, p));
                    vacc3 = vfmaq(vacc3, vx2, vld1q_z(w_base3 + 2 * in_c + ic, p));
                }
                if (fold)
                {
                    arm_nn_f16_fold_pairs_f32(&vacc0_pairs, vacc0, ic0 == 0);
                    vacc0 = vdupq_n_f16((float16_t)0.0f);
                    arm_nn_f16_fold_pairs_f32(&vacc1_pairs, vacc1, ic0 == 0);
                    vacc1 = vdupq_n_f16((float16_t)0.0f);
                    arm_nn_f16_fold_pairs_f32(&vacc2_pairs, vacc2, ic0 == 0);
                    vacc2 = vdupq_n_f16((float16_t)0.0f);
                    arm_nn_f16_fold_pairs_f32(&vacc3_pairs, vacc3, ic0 == 0);
                    vacc3 = vdupq_n_f16((float16_t)0.0f);
                }
                ic0 = ic_end;
            } while (ic0 < in_c);

            const _Float16 acc0 = fold
                ? (_Float16)((b ? (float32_t)b[oc + 0] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc0_pairs))
                : (b ? (_Float16)b[oc + 0] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc0);
            const _Float16 acc1 = fold
                ? (_Float16)((b ? (float32_t)b[oc + 1] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc1_pairs))
                : (b ? (_Float16)b[oc + 1] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc1);
            const _Float16 acc2 = fold
                ? (_Float16)((b ? (float32_t)b[oc + 2] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc2_pairs))
                : (b ? (_Float16)b[oc + 2] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc2);
            const _Float16 acc3 = fold
                ? (_Float16)((b ? (float32_t)b[oc + 3] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc3_pairs))
                : (b ? (_Float16)b[oc + 3] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc3);

            y[oc + 0] = (float16_t)acc0;
            y[oc + 1] = (float16_t)acc1;
            y[oc + 2] = (float16_t)acc2;
            y[oc + 3] = (float16_t)acc3;
        }

        for (; oc < out_c; ++oc)
        {
            const float16_t *w0 = kernel + (size_t)oc * 3U * (size_t)in_c;
            const float16_t *w1 = w0 + in_c;
            const float16_t *w2 = w1 + in_c;
            _Float16 acc = b ? (_Float16)b[oc] : (_Float16)0.0f;
            float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
            float32x4_t vacc_pairs = vdupq_n_f32(0.0f);

            int32_t ic0 = 0;
            do
            {
                const int32_t ic_end = (in_c - ic0 > span) ? ic0 + span : in_c;
                for (int32_t ic = ic0; ic < ic_end; ic += 8)
                {
                    const mve_pred16_t p = vctp16q((uint32_t)(ic_end - ic));
                    vacc = vfmaq(vacc, vld1q_z(x0 + ic, p), vld1q_z(w0 + ic, p));
                    vacc = vfmaq(vacc, vld1q_z(x1 + ic, p), vld1q_z(w1 + ic, p));
                    vacc = vfmaq(vacc, vld1q_z(x2 + ic, p), vld1q_z(w2 + ic, p));
                }
                if (fold)
                {
                    arm_nn_f16_fold_pairs_f32(&vacc_pairs, vacc, ic0 == 0);
                    vacc = vdupq_n_f16((float16_t)0.0f);
                }
                ic0 = ic_end;
            } while (ic0 < in_c);

            if (fold)
            {
                acc = (_Float16)((b ? (float32_t)b[oc] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc_pairs));
            }
            else
            {
                acc += (_Float16)arm_nn_vec_reduce_add_f16(vacc);
            }
            y[oc] = (float16_t)acc;
        }
    #else
        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const float16_t *w = kernel + (size_t)oc * 3U * (size_t)in_c;
            float32_t acc = b ? (float32_t)b[oc] : 0.0f;
            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                acc += (float32_t)x0[ic] * (float32_t)w[ic];
                acc += (float32_t)x1[ic] * (float32_t)w[in_c + ic];
                acc += (float32_t)x2[ic] * (float32_t)w[2 * in_c + ic];
            }
            y[oc] = (float16_t)acc;
        }
    #endif
    }
}

void arm_nn_conv1d_k3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w)
{
    arm_nn_conv1d_k3_nhwc_f16_body(x_nhwc, in_c, in_w, kernel, b, out, out_c, out_w, ARM_NN_F16_ACC_BLOCK);
}

void arm_nn_conv1d_k3_nhwc_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                     int32_t in_c,
                                     int32_t in_w,
                                     const float16_t *__RESTRICT kernel,
                                     const float16_t *__RESTRICT b,
                                     float16_t *__RESTRICT out,
                                     int32_t out_c,
                                     int32_t out_w)
{
    arm_nn_conv1d_k3_nhwc_f16_body(x_nhwc, in_c, in_w, kernel, b, out, out_c, out_w, ARM_NN_F16_ACC_BLOCK_NONE);
}

#endif /* ARM_NN_ENABLE_F16 */
