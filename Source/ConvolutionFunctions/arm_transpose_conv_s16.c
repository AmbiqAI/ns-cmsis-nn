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
 * Title:        arm_transpose_conv_s16.c
 * Description:  s16 transpose convolution with int8 weights and int64 bias.
 *
 * $Date:        8 October 2026
 * $Revision:    V.1.0.0
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

/* TFLite int64 requantization, then activation clamp. */
static int16_t tconv_requant_s16(const int64_t acc,
                                 const int32_t multiplier,
                                 const int32_t shift,
                                 const int32_t act_min,
                                 const int32_t act_max)
{
    const int64_t reduced = REDUCE_MULTIPLIER(multiplier);
    const int32_t total_shift = 15 - shift;
    int64_t result = (acc * reduced + ((int64_t)1 << (total_shift - 1))) >> total_shift;

    result = ARM_NN_MAX(result, (int64_t)act_min);
    result = ARM_NN_MIN(result, (int64_t)act_max);
    return (int16_t)result;
}

/*
 * Basic s16 transpose convolution function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int64_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int16_t *output_data)
{
    (void)ctx;
    (void)output_ctx;
    (void)bias_dims;

    const int32_t stride_x = transpose_conv_params->stride.w;
    const int32_t stride_y = transpose_conv_params->stride.h;

    if (stride_x <= 0 || stride_y <= 0 || transpose_conv_params->dilation.w != 1 ||
        transpose_conv_params->dilation.h != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batches = input_dims->n;
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t input_ch = input_dims->c;

    const int32_t filter_y = filter_dims->h;
    const int32_t filter_x = filter_dims->w;

    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_ch = output_dims->c;

    const int32_t pad_x = transpose_conv_params->padding.w;
    const int32_t pad_y = transpose_conv_params->padding.h;

    const int32_t act_min = transpose_conv_params->activation.min;
    const int32_t act_max = transpose_conv_params->activation.max;

    int16_t *out = output_data;

    for (int32_t b = 0; b < batches; b++)
    {
        const int16_t *batch_in = input_data + (int64_t)b * input_y * input_x * input_ch;

        for (int32_t oy = 0; oy < output_y; oy++)
        {
            for (int32_t ox = 0; ox < output_x; ox++)
            {
                for (int32_t oc = 0; oc < output_ch; oc++)
                {
                    int64_t acc = 0;

                    for (int32_t ky = 0; ky < filter_y; ky++)
                    {
                        /* Find the input row hitting oy. */
                        const int32_t ty = oy + pad_y - ky;
                        if (ty < 0 || ty % stride_y != 0 || ty / stride_y >= input_y)
                        {
                            continue;
                        }
                        const int32_t iy = ty / stride_y;

                        for (int32_t kx = 0; kx < filter_x; kx++)
                        {
                            const int32_t tx = ox + pad_x - kx;
                            if (tx < 0 || tx % stride_x != 0 || tx / stride_x >= input_x)
                            {
                                continue;
                            }
                            const int32_t ix = tx / stride_x;

                            const int16_t *in = batch_in + ((int64_t)iy * input_x + ix) * input_ch;
                            const int8_t *w = filter_data + (((int64_t)oc * filter_y + ky) * filter_x + kx) * input_ch;

                            for (int32_t ic = 0; ic < input_ch; ic++)
                            {
                                acc += (int32_t)in[ic] * (int32_t)w[ic];
                            }
                        }
                    }

                    if (bias_data)
                    {
                        acc += bias_data[oc];
                    }

                    *out++ =
                        tconv_requant_s16(acc, quant_params->multiplier[oc], quant_params->shift[oc], act_min, act_max);
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
