/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_s16.c
 * Description:  s16 version of convolution.
 *
 * $Date:        22 April 2024
 * $Revision:    V.4.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)

/*
 * A 1x1 convolution reduces over input channels only, so when the input pixel
 * fits in a few vectors it can stay in registers while the weights stream past
 * it, instead of the generic matmul re-reading it once per output channel.
 *
 * The vector count is a compile-time constant in each specialisation: with a
 * runtime count the compiler cannot prove how many vectors are live, spills the
 * pixel to the stack and puts a branch in the innermost loop, which costs more
 * than the reloads it saves.
 *
 * Four channels are accumulated before the epilogue so arm_requantize_mve_32x4
 * requantises them in one pass with per-lane multipliers and shifts.
 *
 * int32 accumulator cannot overflow: |32767 * 127| * 32 < 2^31.
 */
    #define CONV_1X1_S16_RESIDENT(NVEC)                                                                                \
        for (int32_t p = 0; p < pixels; p++)                                                                           \
        {                                                                                                              \
            const int16_t *row = input + p * in_ch;                                                                    \
            int16_t *dst = output + p * out_ch;                                                                        \
            const int16x8_t x0 = vld1q_s16(row);                                                                       \
            const int16x8_t x1 = (NVEC > 1) ? vld1q_s16(row + 8) : vdupq_n_s16(0);                                     \
            const int16x8_t x2 = (NVEC > 2) ? vld1q_s16(row + 16) : vdupq_n_s16(0);                                    \
            const int16x8_t x3 = (NVEC > 3) ? vld1q_s16(row + 24) : vdupq_n_s16(0);                                    \
            int32_t c = 0;                                                                                             \
            for (; c + 4 <= out_ch; c += 4)                                                                            \
            {                                                                                                          \
                int32_t acc[4];                                                                                        \
                for (int32_t k = 0; k < 4; k++)                                                                        \
                {                                                                                                      \
                    const int8_t *w = weights + (c + k) * in_ch;                                                       \
                    int32_t a = bias[c + k];                                                                           \
                    a = vmladavaq_s16(a, vldrbq_s16(w), x0);                                                           \
                    if (NVEC > 1)                                                                                      \
                        a = vmladavaq_s16(a, vldrbq_s16(w + 8), x1);                                                   \
                    if (NVEC > 2)                                                                                      \
                        a = vmladavaq_s16(a, vldrbq_s16(w + 16), x2);                                                  \
                    if (NVEC > 3)                                                                                      \
                        a = vmladavaq_s16(a, vldrbq_s16(w + 24), x3);                                                  \
                    acc[k] = a;                                                                                        \
                }                                                                                                      \
                int32x4_t r =                                                                                          \
                    arm_requantize_mve_32x4(vldrwq_s32(acc), vldrwq_s32(out_mult + c), vldrwq_s32(out_shift + c));     \
                r = vminq_s32(vmaxq_s32(r, vmin), vmax);                                                               \
                vstrhq_s32(dst + c, r);                                                                                \
            }                                                                                                          \
            for (; c < out_ch; c++)                                                                                    \
            {                                                                                                          \
                const int8_t *w = weights + c * in_ch;                                                                 \
                int32_t a = bias[c];                                                                                   \
                a = vmladavaq_s16(a, vldrbq_s16(w), x0);                                                               \
                if (NVEC > 1)                                                                                          \
                    a = vmladavaq_s16(a, vldrbq_s16(w + 8), x1);                                                       \
                if (NVEC > 2)                                                                                          \
                    a = vmladavaq_s16(a, vldrbq_s16(w + 16), x2);                                                      \
                if (NVEC > 3)                                                                                          \
                    a = vmladavaq_s16(a, vldrbq_s16(w + 24), x3);                                                      \
                a = arm_nn_requantize(a, out_mult[c], out_shift[c]);                                                   \
                dst[c] = (int16_t)ARM_NN_MIN(ARM_NN_MAX(a, act_min), act_max);                                         \
            }                                                                                                          \
        }

static void conv_1x1_s16_resident_pixel(const int16_t *input,
                                        const int8_t *weights,
                                        const int32_t *bias,
                                        int16_t *output,
                                        const int32_t *out_mult,
                                        const int32_t *out_shift,
                                        const int32_t pixels,
                                        const int32_t out_ch,
                                        const int32_t in_ch,
                                        const int32_t act_min,
                                        const int32_t act_max)
{
    const int32x4_t vmin = vdupq_n_s32(act_min);
    const int32x4_t vmax = vdupq_n_s32(act_max);

    switch (in_ch >> 3)
    {
    case 4:
        CONV_1X1_S16_RESIDENT(4)
        break;
    case 3:
        CONV_1X1_S16_RESIDENT(3)
        break;
    case 2:
        CONV_1X1_S16_RESIDENT(2)
        break;
    default:
        CONV_1X1_S16_RESIDENT(1)
        break;
    }
}

    #undef CONV_1X1_S16_RESIDENT

/* Beyond four vectors the pixel no longer fits alongside a streaming weight
   vector, and a partial vector would need predication in the innermost loop. */
    #define CONV_1X1_S16_RESIDENT_MAX_CH 32

#endif // ARM_MATH_MVEI && !ARM_MATH_AUTOVECTORIZE

/*
 * Pointwise s16 convolution function: no stride, no padding, no dilation.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */

arm_cmsis_nn_status arm_convolve_1x1_s16_ns_np_nd(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_conv_params *conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int16_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const cmsis_nn_bias_data *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int16_t *output_data)
{
    (void)ctx;
    (void)bias_dims;

    const int32_t input_batches = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t rhs_cols = input_ch * kernel_y * kernel_x;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    int32_t lhs_rows = output_x * output_y;

    if (output_data == NULL)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
        if (bias_data != NULL && bias_data->data != NULL && bias_data->is_int32_bias &&
            rhs_cols <= CONV_1X1_S16_RESIDENT_MAX_CH && (rhs_cols & 0x7) == 0)
        {
            conv_1x1_s16_resident_pixel(input_data,
                                        filter_data,
                                        (const int32_t *)bias_data->data,
                                        output_data,
                                        output_mult,
                                        output_shift,
                                        lhs_rows,
                                        output_ch,
                                        rhs_cols,
                                        out_activation_min,
                                        out_activation_max);
        }
        else
#endif
        {
            arm_nn_mat_mult_nt_t_s16(input_data,
                                     filter_data,
                                     bias_data,
                                     output_data,
                                     output_mult,
                                     output_shift,
                                     lhs_rows,
                                     output_dims->c,
                                     rhs_cols,
                                     out_activation_min,
                                     out_activation_max,
                                     output_dims->c);
        }

        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
