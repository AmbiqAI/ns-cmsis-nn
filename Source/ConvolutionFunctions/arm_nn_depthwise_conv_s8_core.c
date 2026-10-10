/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_depthwise_conv_s8_core.c
 * Description:  Depthwise convolution on im2col buffers.
 *
 * $Date:        26 October 2022
 * $Revision:    V.1.0.5
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/*
 * Depthwise conv on an im2col buffer where the input channel equals
 * output channel.
 *
 * Refer header file for details.
 *
 */

int8_t *arm_nn_depthwise_conv_s8_core(const int8_t *row,
                                      const int16_t *col,
                                      const uint16_t num_ch,
                                      const int32_t *out_shift,
                                      const int32_t *out_mult,
                                      const int32_t out_offset,
                                      const int32_t activation_min,
                                      const int32_t activation_max,
                                      const uint16_t kernel_size,
                                      const int32_t *const output_bias,
                                      int8_t *out)
{
#if defined(ARM_MATH_MVEI)
    int32_t ch_per_loop = num_ch / 4;

    const int32_t *bias = output_bias;
    int8_t *out_tmp = out;

    int32_t idx = 0;

    while (ch_per_loop > 0)
    {
        int32x4_t ip_0;
        int32x4_t ip_1;
        int32_t ker_loop = kernel_size / 3;
        int32x4_t out_0 = vldrwq_s32(bias);
        int32x4_t out_1 = out_0;
        bias += 4;

        const int32_t offset = idx * 4;
        const int8_t *row_0 = row + offset;
        const int16_t *col_0 = col + offset;
        const int16_t *col_1 = col + kernel_size * num_ch + offset;

        int32x4_t ker_0 = vldrbq_s32(row_0);

        while (ker_loop > 0)
        {
            const int8_t *row_1 = row_0 + num_ch;
            const int8_t *row_2 = row_0 + 2 * num_ch;
            const int32x4_t ker_1 = vldrbq_s32(row_1);
            const int32x4_t ker_2 = vldrbq_s32(row_2);

            ip_0 = vldrhq_s32(col_0);
            ip_1 = vldrhq_s32(col_1);
            col_0 += num_ch;
            col_1 += num_ch;

            out_0 += vmulq_s32(ip_0, ker_0);
            out_1 += vmulq_s32(ip_1, ker_0);

            ip_0 = vldrhq_s32(col_0);
            ip_1 = vldrhq_s32(col_1);
            col_0 += num_ch;
            col_1 += num_ch;

            out_0 += vmulq_s32(ip_0, ker_1);
            out_1 += vmulq_s32(ip_1, ker_1);

            ip_0 = vldrhq_s32(col_0);
            ip_1 = vldrhq_s32(col_1);
            col_0 += num_ch;
            col_1 += num_ch;

            out_0 += vmulq_s32(ip_0, ker_2);
            out_1 += vmulq_s32(ip_1, ker_2);
            row_0 += 3 * num_ch;

            ker_0 = vldrbq_s32(row_0);
            ker_loop--;
        }

        idx++;
        /* Handle tail kernel elements */
        ker_loop = kernel_size - ((kernel_size / 3) * 3);
        while (ker_loop > 0)
        {
            ip_0 = vldrhq_s32(col_0);
            ip_1 = vldrhq_s32(col_1);

            out_0 += vmulq_s32(ip_0, ker_0);
            out_1 += vmulq_s32(ip_1, ker_0);

            col_0 += num_ch;
            col_1 += num_ch;

            ip_0 = vldrhq_s32(col_0);
            ip_1 = vldrhq_s32(col_1);

            row_0 += num_ch;
            ker_0 = vldrbq_s32(row_0);
            ker_loop--;
        }
        const int32x4_t mult = vldrwq_s32(out_mult);
        const int32x4_t shift = vldrwq_s32(out_shift);
        out_mult += 4;
        out_shift += 4;

        out_0 = arm_requantize_mve_32x4(out_0, mult, shift);
        out_1 = arm_requantize_mve_32x4(out_1, mult, shift);

        out_0 = vaddq_n_s32(out_0, out_offset);
        out_0 = vmaxq_s32(out_0, vdupq_n_s32(activation_min));
        out_0 = vminq_s32(out_0, vdupq_n_s32(activation_max));
        vstrbq_s32(out_tmp, out_0);

        out_1 = vaddq_n_s32(out_1, out_offset);
        out_1 = vmaxq_s32(out_1, vdupq_n_s32(activation_min));
        out_1 = vminq_s32(out_1, vdupq_n_s32(activation_max));
        vstrbq_s32(out_tmp + num_ch, out_1);

        out_tmp += 4;
        ch_per_loop--;
    }

    int32_t tail_ch = num_ch & 3;
    if (tail_ch != 0)
    {
        int32_t ch_idx = (num_ch & ~3);
        int32x4_t col_0_sum;
        int32x4_t col_1_sum;

        const int32_t single_buffer_size = kernel_size * num_ch;
        for (int i = 0; i < tail_ch; i++)
        {
            const int16_t *col_pos_0 = col + ch_idx;
            const int16_t *col_pos_1 = col_pos_0 + single_buffer_size;

            const int8_t *row_pos = row + ch_idx;
            int32_t sum_0 = bias[i];
            int32_t sum_1 = bias[i];

            for (int j = 0; j < kernel_size; j++)
            {
                const int8_t row_val = row_pos[j * num_ch];
                sum_0 += row_val * col_pos_0[j * num_ch];
                sum_1 += row_val * col_pos_1[j * num_ch];
            }
            col_0_sum[i] = sum_0;
            col_1_sum[i] = sum_1;

            ch_idx++;
        }
        const mve_pred16_t p = vctp32q((uint32_t)tail_ch);
        const int32x4_t mult = vldrwq_z_s32(out_mult, p);
        const int32x4_t shift = vldrwq_z_s32(out_shift, p);

        col_0_sum = arm_requantize_mve_32x4(col_0_sum, mult, shift);
        col_1_sum = arm_requantize_mve_32x4(col_1_sum, mult, shift);

        col_0_sum = vaddq_n_s32(col_0_sum, out_offset);
        col_0_sum = vmaxq_s32(col_0_sum, vdupq_n_s32(activation_min));
        col_0_sum = vminq_s32(col_0_sum, vdupq_n_s32(activation_max));
        vstrbq_p_s32(out_tmp, col_0_sum, p);

        col_1_sum = vaddq_n_s32(col_1_sum, out_offset);
        col_1_sum = vmaxq_s32(col_1_sum, vdupq_n_s32(activation_min));
        col_1_sum = vminq_s32(col_1_sum, vdupq_n_s32(activation_max));
        vstrbq_p_s32(out_tmp + num_ch, col_1_sum, p);

        out_tmp += tail_ch;
    }

    return out_tmp + num_ch;
#else
    (void)row;
    (void)col;
    (void)num_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)kernel_size;
    (void)output_bias;
    (void)out;
    return NULL;
#endif
}

#if defined(ARM_MATH_MVEI)
/* Tap range of one output along an axis */
__STATIC_FORCEINLINE void dw_tap_range(const int32_t base,
                                       const int32_t in_size,
                                       const int32_t ker_size,
                                       const int32_t dilation,
                                       int32_t *start,
                                       int32_t *end)
{
    if (dilation > 1)
    {
        *start = ARM_NN_MAX(0, (-base + dilation - 1) / dilation);
        *end = ARM_NN_MIN(ker_size, (in_size - base + dilation - 1) / dilation);
    }
    else
    {
        *start = ARM_NN_MAX(0, -base);
        *end = ARM_NN_MIN(ker_size, in_size - base);
    }
}

/* Input load modes of the MVE kernel */
enum
{
    DW_LOAD_CONTIGUOUS,
    DW_LOAD_GATHER,
    DW_LOAD_BROADCAST
};

/* Adds one tap for four output channels */
__STATIC_FORCEINLINE int32x4_t dw_mac(int32x4_t acc,
                                      const int8_t *px,
                                      const int8_t *ker,
                                      const int32_t oc,
                                      const uint32x4_t offs,
                                      const int32_t idx,
                                      const mve_pred16_t p,
                                      const int32_t input_offset,
                                      const int32_t mode)
{
    const int32x4_t k = vldrbq_z_s32(ker, p);
    if (mode == DW_LOAD_BROADCAST)
    {
        return vmlaq_n_s32(acc, k, px[idx] + input_offset);
    }
    int32x4_t ip;
    if (mode == DW_LOAD_CONTIGUOUS)
    {
        ip = vldrbq_z_s32(px + oc, p);
    }
    else
    {
        ip = vldrbq_gather_offset_z_s32(px, offs, p);
    }
    ip = vaddq_n_s32(ip, input_offset);
    return vaddq_s32(acc, vmulq_s32(ip, k));
}

/* Requantizes and stores four output channels */
__STATIC_FORCEINLINE void dw_store_4(int8_t *out,
                                     int32x4_t acc,
                                     const int32x4_t mult,
                                     const int32x4_t shift,
                                     const int32_t output_offset,
                                     const int32_t act_min,
                                     const int32_t act_max,
                                     const mve_pred16_t p)
{
    acc = arm_requantize_mve_32x4(acc, mult, shift);
    acc = vaddq_n_s32(acc, output_offset);
    acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
    acc = vminq_s32(acc, vdupq_n_s32(act_max));
    vstrbq_p_s32(out, acc, p);
}

/* Channel-vectorized depthwise for any multiplier */
__STATIC_FORCEINLINE void dw_conv_s8_mve_impl(const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *kernel,
                                              const int32_t *bias,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output,
                                              const int32_t ch_mult,
                                              const int32_t mode)
{
    const int32_t input_batches = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t dilation_x = dw_conv_params->dilation.w;
    const int32_t dilation_y = dw_conv_params->dilation.h;
    const int32_t output_offset = dw_conv_params->output_offset;
    const int32_t input_offset = dw_conv_params->input_offset;
    const int32_t act_min = dw_conv_params->activation.min;
    const int32_t act_max = dw_conv_params->activation.max;
    const int32_t *output_mult = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    const int32_t ker_row_step = kernel_x * output_ch;
    const int32_t in_row_step = dilation_y * input_x * input_ch;
    const int32_t in_tap_step = dilation_x * input_ch;

    const int8_t *in_batch = input;
    int8_t *out_row = output;

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++, in_batch += input_x * input_y * input_ch)
    {
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++, out_row += output_x * output_ch)
        {
            const int32_t base_y = i_out_y * stride_y - pad_y;
            int32_t ky_start, ky_end;
            dw_tap_range(base_y, input_y, kernel_y, dilation_y, &ky_start, &ky_end);

            /* Blocks of four channels per row */
            for (int32_t oc = 0; oc < output_ch; oc += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(output_ch - oc));
                const int32_t idx = oc / ch_mult;
                uint32_t lane_offs[4];
                for (int32_t l = 0; l < 4; l++)
                {
                    lane_offs[l] = (uint32_t)((oc + l) / ch_mult);
                }
                const uint32x4_t offs = vldrwq_u32(lane_offs);
                const int32x4_t mult = vldrwq_z_s32(output_mult + oc, p);
                const int32x4_t shift = vldrwq_z_s32(output_shift + oc, p);
                const int32x4_t init = bias ? vldrwq_z_s32(bias + oc, p) : vdupq_n_s32(0);
                int8_t *out = out_row + oc;

                for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++, out += output_ch)
                {
                    const int32_t base_x = i_out_x * stride_x - pad_x;
                    int32_t kx_start, kx_end;
                    dw_tap_range(base_x, input_x, kernel_x, dilation_x, &kx_start, &kx_end);

                    int32x4_t acc = init;
                    const int8_t *in_row = in_batch +
                        ((base_y + ky_start * dilation_y) * input_x + base_x + kx_start * dilation_x) * input_ch;
                    const int8_t *ker_row = kernel + (ky_start * kernel_x + kx_start) * output_ch + oc;

                    for (int32_t ky = ky_start; ky < ky_end; ky++, in_row += in_row_step, ker_row += ker_row_step)
                    {
                        const int8_t *px = in_row;
                        const int8_t *ker = ker_row;
                        for (int32_t kx = kx_start; kx < kx_end; kx++, px += in_tap_step, ker += output_ch)
                        {
                            acc = dw_mac(acc, px, ker, oc, offs, idx, p, input_offset, mode);
                        }
                    }

                    dw_store_4(out, acc, mult, shift, output_offset, act_min, act_max, p);
                }
            }
        }
    }
}
#endif

/*
 * Channel-vectorized s8 depthwise convolution for any layer.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_s8_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                 const cmsis_nn_per_channel_quant_params *quant_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const int8_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const int8_t *kernel,
                                                 const int32_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 int8_t *output)
{
#if defined(ARM_MATH_MVEI)
    const int32_t ch_mult = dw_conv_params->ch_mult;
    if (ch_mult == 1)
    {
        dw_conv_s8_mve_impl(dw_conv_params,
                            quant_params,
                            input_dims,
                            input,
                            filter_dims,
                            kernel,
                            bias,
                            output_dims,
                            output,
                            1,
                            DW_LOAD_CONTIGUOUS);
    }
    else if (ch_mult % 4 == 0)
    {
        dw_conv_s8_mve_impl(dw_conv_params,
                            quant_params,
                            input_dims,
                            input,
                            filter_dims,
                            kernel,
                            bias,
                            output_dims,
                            output,
                            ch_mult,
                            DW_LOAD_BROADCAST);
    }
    else
    {
        dw_conv_s8_mve_impl(dw_conv_params,
                            quant_params,
                            input_dims,
                            input,
                            filter_dims,
                            kernel,
                            bias,
                            output_dims,
                            output,
                            ch_mult,
                            DW_LOAD_GATHER);
    }
    return ARM_CMSIS_NN_SUCCESS;
#else
    (void)dw_conv_params;
    (void)quant_params;
    (void)input_dims;
    (void)input;
    (void)filter_dims;
    (void)kernel;
    (void)bias;
    (void)output_dims;
    (void)output;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}
