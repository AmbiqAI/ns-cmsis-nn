/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k5_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 5 for f16
 *
 * $Date:        31 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV1D_K5_NHWC_F16_OC4_BLOCK (4)
        #define ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK (16)
    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
/* One 8-channel step of four output channels: five taps each, the loads predicated by p. */
__STATIC_FORCEINLINE void arm_nn_conv1d_k5_oc4_step_f16(const float16_t *const x[5],
                                                        const float16_t *const w[4],
                                                        const int32_t in_c,
                                                        const int32_t ic,
                                                        const mve_pred16_t p,
                                                        float16x8_t acc[4])
{
    const float16x8_t vx0 = vld1q_z(x[0] + ic, p);
    const float16x8_t vx1 = vld1q_z(x[1] + ic, p);
    const float16x8_t vx2 = vld1q_z(x[2] + ic, p);
    const float16x8_t vx3 = vld1q_z(x[3] + ic, p);
    const float16x8_t vx4 = vld1q_z(x[4] + ic, p);
    for (int32_t j = 0; j < 4; ++j)
    {
        acc[j] = vfmaq(acc[j], vx0, vld1q_z(w[j] + 0 * in_c + ic, p));
        acc[j] = vfmaq(acc[j], vx1, vld1q_z(w[j] + 1 * in_c + ic, p));
        acc[j] = vfmaq(acc[j], vx2, vld1q_z(w[j] + 2 * in_c + ic, p));
        acc[j] = vfmaq(acc[j], vx3, vld1q_z(w[j] + 3 * in_c + ic, p));
        acc[j] = vfmaq(acc[j], vx4, vld1q_z(w[j] + 4 * in_c + ic, p));
    }
}
    #endif

__STATIC_FORCEINLINE void arm_nn_conv1d_k5_nhwc_f16_body(const float16_t *__RESTRICT x_nhwc,
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
    /* Each lane takes 5 taps per input-channel vector. With more than `block` taps per output, a lane's float16
     * partial covers at most block / 5 channel vectors; each block's lanes are folded into float32 pair
     * accumulators (arm_nn_f16_fold_pairs_f32), summed once (arm_nn_f16_pairs_sum_f32), the bias is added in float32
     * and the total rounds to float16 once (#586). */
    const bool fold = 5 * in_c > block;
    const int32_t span = fold ? (block / 5) * 8 : in_c;
    #endif

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t oc = 0;

        if (in_c == 16 && b != NULL)
        {
            const float16x8_t vx00 = vld1q(x0 + 0);
            const float16x8_t vx01 = vld1q(x1 + 0);
            const float16x8_t vx02 = vld1q(x2 + 0);
            const float16x8_t vx03 = vld1q(x3 + 0);
            const float16x8_t vx04 = vld1q(x4 + 0);
            const float16x8_t vx10 = vld1q(x0 + 8);
            const float16x8_t vx11 = vld1q(x1 + 8);
            const float16x8_t vx12 = vld1q(x2 + 8);
            const float16x8_t vx13 = vld1q(x3 + 8);
            const float16x8_t vx14 = vld1q(x4 + 8);

            for (; oc + ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK <= out_c; oc += ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK)
            {
                float16x8_t partials[ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK];

                for (int32_t of = 0; of < ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK; ++of)
                {
                    const float16_t *w_base = kernel + (size_t)(oc + of) * 5U * (size_t)in_c;
                    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);

                    vacc = vfmaq(vacc, vx00, vld1q(w_base + 0 * in_c + 0));
                    vacc = vfmaq(vacc, vx01, vld1q(w_base + 1 * in_c + 0));
                    vacc = vfmaq(vacc, vx02, vld1q(w_base + 2 * in_c + 0));
                    vacc = vfmaq(vacc, vx03, vld1q(w_base + 3 * in_c + 0));
                    vacc = vfmaq(vacc, vx04, vld1q(w_base + 4 * in_c + 0));
                    partials[of] = vacc;
                }

                for (int32_t of = 0; of < ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK; ++of)
                {
                    const float16_t *w_base = kernel + (size_t)(oc + of) * 5U * (size_t)in_c + 8;
                    float16x8_t vacc = partials[of];

                    vacc = vfmaq(vacc, vx10, vld1q(w_base + 0 * in_c));
                    vacc = vfmaq(vacc, vx11, vld1q(w_base + 1 * in_c));
                    vacc = vfmaq(vacc, vx12, vld1q(w_base + 2 * in_c));
                    vacc = vfmaq(vacc, vx13, vld1q(w_base + 3 * in_c));
                    vacc = vfmaq(vacc, vx14, vld1q(w_base + 4 * in_c));

                    /* 80 taps, ten per lane: one block, summed in float32 when folding. */
                    if (fold)
                    {
                        y[oc + of] = (float16_t)((float32_t)b[oc + of] + arm_nn_vec_reduce_add_f16_to_f32(vacc));
                    }
                    else
                    {
                        y[oc + of] = (float16_t)((_Float16)b[oc + of] + (_Float16)arm_nn_vec_reduce_add_f16(vacc));
                    }
                }
            }
        }

        for (; oc + ARM_NN_CONV1D_K5_NHWC_F16_OC4_BLOCK <= out_c; oc += ARM_NN_CONV1D_K5_NHWC_F16_OC4_BLOCK)
        {
            const float16_t *w_base0 = kernel + (size_t)(oc + 0) * 5U * (size_t)in_c;
            const float16_t *w_base1 = kernel + (size_t)(oc + 1) * 5U * (size_t)in_c;
            const float16_t *w_base2 = kernel + (size_t)(oc + 2) * 5U * (size_t)in_c;
            const float16_t *w_base3 = kernel + (size_t)(oc + 3) * 5U * (size_t)in_c;
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
                /* Whole 8-channel steps, then one predicated tail step outside the loop: GCC 14.3 at -Ofast turned
                   a tail-predicated loop here into dlstp with full-register accumulator moves in its body, which
                   under tail predication dropped the inactive lanes of an accumulator on the last step (#588). */
                const float16_t *const xs[5] = {x0, x1, x2, x3, x4};
                const float16_t *const ws[4] = {w_base0, w_base1, w_base2, w_base3};
                float16x8_t acc[4] = {vacc0, vacc1, vacc2, vacc3};
                int32_t ic = ic0;
                for (; ic + 8 <= ic_end; ic += 8)
                {
                    arm_nn_conv1d_k5_oc4_step_f16(xs, ws, in_c, ic, (mve_pred16_t)0xFFFF, acc);
                }
                if (ic < ic_end)
                {
                    arm_nn_conv1d_k5_oc4_step_f16(xs, ws, in_c, ic, vctp16q((uint32_t)(ic_end - ic)), acc);
                }
                vacc0 = acc[0];
                vacc1 = acc[1];
                vacc2 = acc[2];
                vacc3 = acc[3];
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

            if (fold)
            {
                y[oc + 0] =
                    (float16_t)(_Float16)((b ? (float32_t)b[oc + 0] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc0_pairs));
                y[oc + 1] =
                    (float16_t)(_Float16)((b ? (float32_t)b[oc + 1] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc1_pairs));
                y[oc + 2] =
                    (float16_t)(_Float16)((b ? (float32_t)b[oc + 2] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc2_pairs));
                y[oc + 3] =
                    (float16_t)(_Float16)((b ? (float32_t)b[oc + 3] : 0.0f) + arm_nn_f16_pairs_sum_f32(vacc3_pairs));
            }
            else
            {
                /* Each lane sum is pinned before the bias add so -ffast-math keeps the float16-lane order. */
                _Float16 sum0 = (_Float16)arm_nn_vec_reduce_add_f16(vacc0);
                _Float16 sum1 = (_Float16)arm_nn_vec_reduce_add_f16(vacc1);
                _Float16 sum2 = (_Float16)arm_nn_vec_reduce_add_f16(vacc2);
                _Float16 sum3 = (_Float16)arm_nn_vec_reduce_add_f16(vacc3);
                __asm__("" : "+t"(sum0), "+t"(sum1), "+t"(sum2), "+t"(sum3));
                y[oc + 0] = (float16_t)((b ? (_Float16)b[oc + 0] : (_Float16)0.0f) + sum0);
                y[oc + 1] = (float16_t)((b ? (_Float16)b[oc + 1] : (_Float16)0.0f) + sum1);
                y[oc + 2] = (float16_t)((b ? (_Float16)b[oc + 2] : (_Float16)0.0f) + sum2);
                y[oc + 3] = (float16_t)((b ? (_Float16)b[oc + 3] : (_Float16)0.0f) + sum3);
            }
        }

        for (; oc < out_c; ++oc)
        {
            const float16_t *w0 = kernel + (size_t)oc * 5U * (size_t)in_c;
            const float16_t *w1 = w0 + in_c;
            const float16_t *w2 = w1 + in_c;
            const float16_t *w3 = w2 + in_c;
            const float16_t *w4 = w3 + in_c;
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
                    vacc = vfmaq(vacc, vld1q_z(x3 + ic, p), vld1q_z(w3 + ic, p));
                    vacc = vfmaq(vacc, vld1q_z(x4 + ic, p), vld1q_z(w4 + ic, p));
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
            const float16_t *w = kernel + (size_t)oc * 5U * (size_t)in_c;
            float32_t acc = b ? (float32_t)b[oc] : 0.0f;
            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                acc += (float32_t)x0[ic] * (float32_t)w[ic];
                acc += (float32_t)x1[ic] * (float32_t)w[in_c + ic];
                acc += (float32_t)x2[ic] * (float32_t)w[2 * in_c + ic];
                acc += (float32_t)x3[ic] * (float32_t)w[3 * in_c + ic];
                acc += (float32_t)x4[ic] * (float32_t)w[4 * in_c + ic];
            }
            y[oc] = (float16_t)acc;
        }
    #endif
    }
}

void arm_nn_conv1d_k5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w)
{
    arm_nn_conv1d_k5_nhwc_f16_body(x_nhwc, in_c, in_w, kernel, b, out, out_c, out_w, ARM_NN_F16_ACC_BLOCK);
}

void arm_nn_conv1d_k5_nhwc_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                     int32_t in_c,
                                     int32_t in_w,
                                     const float16_t *__RESTRICT kernel,
                                     const float16_t *__RESTRICT b,
                                     float16_t *__RESTRICT out,
                                     int32_t out_c,
                                     int32_t out_w)
{
    arm_nn_conv1d_k5_nhwc_f16_body(x_nhwc, in_c, in_w, kernel, b, out, out_c, out_w, ARM_NN_F16_ACC_BLOCK_NONE);
}

#endif /* ARM_NN_ENABLE_F16 */
