/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_s8_3x3_c16_s1.c
 * Description:  Direct-entry s8 3x3 convolution over 16 input channels with unit stride, reading the input in place.
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
 * 3x3 kernel over a 16-deep input with unit stride and dilation, one group and no upscale. A pixel's 144-value patch
 * is three 48-byte kernel rows. For a patch inside the input those rows are read in place; a patch that crosses the
 * border is first copied to an im2col slot with padding. Four pixels are then multiplied against every filter as in
 * arm_nn_mat_mult_nt_t_s8(), with the same requantization, so the output is the same bit for bit.
 */
__STATIC_FORCEINLINE int arm_convolve_s8_is_3x3_c16_s1(const cmsis_nn_conv_params *conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *upscale_dims)
{
    return (upscale_dims == NULL) && (input_dims->c == 16) && (filter_dims->c == 16) && (filter_dims->w == 3) &&
        (filter_dims->h == 3) && (conv_params->stride.w == 1) && (conv_params->stride.h == 1) &&
        (conv_params->dilation.w == 1) && (conv_params->dilation.h == 1);
}

/* Output channels for four pixels whose kernel rows start at seg[ky][pixel]; only the first n_pix are stored. */
static __attribute__((noinline)) void arm_convolve_s8_3x3_c16_block(const int8_t *seg[3][4],
                                                                    const int32_t n_pix,
                                                                    const int8_t *rhs,
                                                                    const int32_t *weight_sum,
                                                                    int8_t *dst,
                                                                    const int32_t *dst_multipliers,
                                                                    const int32_t *dst_shifts,
                                                                    const int32_t rhs_rows,
                                                                    const int32_t dst_offset,
                                                                    const int32_t activation_min,
                                                                    const int32_t activation_max)
{
    const uint32x4_t scatter = vmulq_n_u32(vidupq_n_u32(0, 1), (uint32_t)rhs_rows);
    const mve_pred16_t p_store = vctp32q((uint32_t)n_pix);
    const int32x4_t v_min = vdupq_n_s32(activation_min);
    const int32x4_t v_max = vdupq_n_s32(activation_max);
    for (int32_t i = 0; i < rhs_rows; i++)
    {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const int8_t *w = rhs + i * 144;
        for (int32_t ky = 0; ky < 3; ky++)
        {
            /* Volatile so the patch loads stay inside the channel loop instead of being hoisted and spilled. */
            __ASM volatile("   vldrb.8         q0, [%[w]], #16        \n"
                           "   vldrb.8         q1, [%[x0]]            \n"
                           "   vmladava.s8     %[a0], q0, q1          \n"
                           "   vldrb.8         q2, [%[x1]]            \n"
                           "   vmladava.s8     %[a1], q0, q2          \n"
                           "   vldrb.8         q3, [%[x2]]            \n"
                           "   vmladava.s8     %[a2], q0, q3          \n"
                           "   vldrb.8         q4, [%[x3]]            \n"
                           "   vmladava.s8     %[a3], q0, q4          \n"
                           "   vldrb.8         q0, [%[w]], #16        \n"
                           "   vldrb.8         q1, [%[x0], #16]       \n"
                           "   vmladava.s8     %[a0], q0, q1          \n"
                           "   vldrb.8         q2, [%[x1], #16]       \n"
                           "   vmladava.s8     %[a1], q0, q2          \n"
                           "   vldrb.8         q3, [%[x2], #16]       \n"
                           "   vmladava.s8     %[a2], q0, q3          \n"
                           "   vldrb.8         q4, [%[x3], #16]       \n"
                           "   vmladava.s8     %[a3], q0, q4          \n"
                           "   vldrb.8         q0, [%[w]], #16        \n"
                           "   vldrb.8         q1, [%[x0], #32]       \n"
                           "   vmladava.s8     %[a0], q0, q1          \n"
                           "   vldrb.8         q2, [%[x1], #32]       \n"
                           "   vmladava.s8     %[a1], q0, q2          \n"
                           "   vldrb.8         q3, [%[x2], #32]       \n"
                           "   vmladava.s8     %[a2], q0, q3          \n"
                           "   vldrb.8         q4, [%[x3], #32]       \n"
                           "   vmladava.s8     %[a3], q0, q4          \n"
                           : [w] "+r"(w), [a0] "+Te"(a0), [a1] "+Te"(a1), [a2] "+Te"(a2), [a3] "+Te"(a3)
                           : [x0] "r"(seg[ky][0]), [x1] "r"(seg[ky][1]), [x2] "r"(seg[ky][2]), [x3] "r"(seg[ky][3])
                           : "q0", "q1", "q2", "q3", "q4", "memory");
        }
        int32x4_t res = {a0, a1, a2, a3};
        res = vaddq_n_s32(res, weight_sum[i]);
        res = arm_requantize_mve(res, dst_multipliers[i], dst_shifts[i]);
        res = vaddq_n_s32(res, dst_offset);
        res = vminq_s32(vmaxq_s32(res, v_min), v_max);
        vstrbq_scatter_offset_p_s32(dst + i, scatter, res, p_store);
    }
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_s8_3x3_c16_s1_kernel(const cmsis_nn_context *ctx,
                                  const cmsis_nn_context *weight_sum_ctx,
                                  const cmsis_nn_conv_params *conv_params,
                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                  const cmsis_nn_dims *input_dims,
                                  const int8_t *input_data,
                                  const int8_t *filter_data,
                                  const cmsis_nn_dims *output_dims,
                                  int8_t *output_data)
{
    if (ctx->buf == NULL || weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int8_t *buf = (int8_t *)ctx->buf;
    const int32_t *weight_sum = (const int32_t *)weight_sum_ctx->buf;

    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t in_row_stride = input_x * 16;
    const int8x16_t pad_val = vdupq_n_s8((int8_t)-conv_params->input_offset);

    for (int32_t i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        const int8_t *seg[3][4];
        int32_t n_pix = 0;
        int8_t *out = output_data;
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int32_t base_y = i_out_y - pad_y;
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_x = i_out_x - pad_x;
                if (base_x >= 0 && base_x + 3 <= input_x && base_y >= 0 && base_y + 3 <= input_y)
                {
                    const int8_t *p = input_data + base_y * in_row_stride + base_x * 16;
                    seg[0][n_pix] = p;
                    seg[1][n_pix] = p + in_row_stride;
                    seg[2][n_pix] = p + 2 * in_row_stride;
                }
                else
                {
                    int8_t *slot = buf + n_pix * 144;
                    int8_t *dst = slot;
                    for (int32_t ky = 0; ky < 3; ky++)
                    {
                        const int32_t k_y = base_y + ky;
                        for (int32_t kx = 0; kx < 3; kx++)
                        {
                            const int32_t k_x = base_x + kx;
                            int8x16_t v = pad_val;
                            if (k_y >= 0 && k_y < input_y && k_x >= 0 && k_x < input_x)
                            {
                                v = vldrbq_s8(input_data + k_y * in_row_stride + k_x * 16);
                            }
                            vstrbq_s8(dst, v);
                            dst += 16;
                        }
                    }
                    seg[0][n_pix] = slot;
                    seg[1][n_pix] = slot + 48;
                    seg[2][n_pix] = slot + 96;
                }
                n_pix++;
                if (n_pix == 4)
                {
                    arm_convolve_s8_3x3_c16_block(seg,
                                                  4,
                                                  filter_data,
                                                  weight_sum,
                                                  out,
                                                  quant_params->multiplier,
                                                  quant_params->shift,
                                                  output_ch,
                                                  conv_params->output_offset,
                                                  conv_params->activation.min,
                                                  conv_params->activation.max);
                    out += 4 * output_ch;
                    n_pix = 0;
                }
            }
        }
        if (n_pix != 0)
        {
            /* Unused lanes repeat the first pixel so every load stays in bounds; they are not stored. */
            for (int32_t j = n_pix; j < 4; j++)
            {
                seg[0][j] = seg[0][0];
                seg[1][j] = seg[1][0];
                seg[2][j] = seg[2][0];
            }
            arm_convolve_s8_3x3_c16_block(seg,
                                          n_pix,
                                          filter_data,
                                          weight_sum,
                                          out,
                                          quant_params->multiplier,
                                          quant_params->shift,
                                          output_ch,
                                          conv_params->output_offset,
                                          conv_params->activation.min,
                                          conv_params->activation.max);
        }
        input_data += input_x * input_y * 16;
        output_data += output_x * output_y * output_ch;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif

/*
 * s8 3x3 convolution over 16 input channels with unit stride and dilation.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s8_3x3_c16_s1(const cmsis_nn_context *ctx,
                                               const cmsis_nn_context *weight_sum_ctx,
                                               const cmsis_nn_conv_params *conv_params,
                                               const cmsis_nn_per_channel_quant_params *quant_params,
                                               const cmsis_nn_dims *input_dims,
                                               const int8_t *input_data,
                                               const cmsis_nn_dims *filter_dims,
                                               const int8_t *filter_data,
                                               const cmsis_nn_dims *bias_dims,
                                               const int32_t *bias_data,
                                               const cmsis_nn_dims *upscale_dims,
                                               const cmsis_nn_dims *output_dims,
                                               int8_t *output_data)
{
    (void)bias_dims;
    (void)bias_data;

    /* The argument checks of arm_convolve_s8() */
    if (ctx->buf == NULL || arm_nn_convolve_s8_groups_invalid(input_dims, filter_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#if defined(ARM_MATH_MVEI)
    if (weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#endif

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (!arm_convolve_s8_is_3x3_c16_s1(conv_params, input_dims, filter_dims, upscale_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    /* The bias is folded into the weight sums */
    return arm_convolve_s8_3x3_c16_s1_kernel(
        ctx, weight_sum_ctx, conv_params, quant_params, input_dims, input_data, filter_data, output_dims, output_data);
#else
    (void)weight_sum_ctx;
    (void)conv_params;
    (void)quant_params;
    (void)input_data;
    (void)filter_data;
    (void)upscale_dims;
    (void)output_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of NNConv group
 */
