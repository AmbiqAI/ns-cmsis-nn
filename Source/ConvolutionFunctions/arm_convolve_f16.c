/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_f16.c
 * Description:  Generic float16 convolution
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

/* Generic float16 convolution. */

#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F16

    #include "Internal/arm_conv_opt_common.h"
    #include "Internal/arm_conv_opt_f16.h"
    #include "Internal/arm_nn_activation_flt.h"
    #include "arm_nnfunctions.h"
    #include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

__STATIC_INLINE bool arm_conv_nhwc_use_patch_gemm_f16(const cmsis_nn_context *ctx,
                                                      int32_t patch_len,
                                                      int32_t output_c,
                                                      int32_t output_positions)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || patch_len <= 0)
    {
        return false;
    }

    /* No patch-length floor: the sizer already covers every patch length (#417). */
    if (output_c < ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_OC || output_positions < ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_POS)
    {
        return false;
    }

    const size_t row_bytes = (size_t)patch_len * sizeof(float16_t);
    return row_bytes > 0U && (size_t)ctx->size >= row_bytes;
}

__STATIC_INLINE bool arm_conv_nhwc_use_1x1_f16(const cmsis_nn_conv_params_f16 *conv_params,
                                               const cmsis_nn_dims *filter_dims)
{
    return conv_params && filter_dims && filter_dims->h == 1 && filter_dims->w == 1 && conv_params->padding.h == 0 &&
        conv_params->padding.w == 0;
}

__STATIC_INLINE arm_cmsis_nn_status arm_convolve_patch_mat_mul_f16(const float16_t *lhs,
                                                                   const float16_t *rhs,
                                                                   const float16_t *bias,
                                                                   float16_t *dst,
                                                                   int32_t lhs_rows,
                                                                   int32_t rhs_rows,
                                                                   int32_t rhs_cols,
                                                                   int32_t row_address_offset,
                                                                   const cmsis_nn_conv_params_f16 *conv_params)
{
    if (conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        return arm_nn_mat_mult_nt_n_packed_f16(lhs,
                                               rhs,
                                               bias,
                                               dst,
                                               lhs_rows,
                                               rhs_rows,
                                               rhs_cols,
                                               row_address_offset,
                                               conv_params->activation.min,
                                               conv_params->activation.max);
    }

    return arm_nn_mat_mult_nt_t_f16(lhs,
                                    rhs,
                                    bias,
                                    dst,
                                    lhs_rows,
                                    rhs_rows,
                                    rhs_cols,
                                    row_address_offset,
                                    conv_params->activation.min,
                                    conv_params->activation.max);
}

__STATIC_INLINE bool arm_conv_nhwc_use_1xn_f16(const cmsis_nn_context *ctx,
                                               const cmsis_nn_conv_params_f16 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || !conv_params || !input_dims || !filter_dims || !output_dims)
    {
        return false;
    }

    /* This helper only selects the generic 1xN NHWC kernel family. */
    if (input_dims->h != 1 || output_dims->h != 1 || filter_dims->h != 1 || filter_dims->w <= 1 ||
        conv_params->stride.h != 1 || conv_params->stride.w <= 0 || conv_params->padding.h != 0 ||
        conv_params->dilation.h != 1 || conv_params->dilation.w != 1)
    {
        return false;
    }

    #ifndef NN_DISABLE_SPECIALIZATION
    /*
     * If a direct specialization already claims the shape, let the normal
     * specialization dispatcher handle it instead of forcing the generic 1xN
     * implementation to know kernel-specific details.
     */
    if (arm_conv_spec_nhwc_f16_matches_any(ctx, conv_params, input_dims, filter_dims, output_dims))
    {
        return false;
    }
    #endif

    /* Remaining 1xN shapes use the generic packed-input helper when workspace is available. */
    const int32_t buf_size =
        arm_convolve_1_x_n_f16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
    return buf_size > 0 && ctx->size >= buf_size;
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV_SMALL_C_F16_LANES (8)
        #define ARM_NN_CONV_SMALL_C_F16_OC_GROUP (4)

/* Direct kernel for fewer input channels than one vector: lanes are output x positions, inputs are
 * gathered per (tap, ic), weights are scalars, and OC_GROUP output channels accumulate at once (#417). */
__STATIC_INLINE bool arm_conv_nhwc_use_small_c_f16(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *output_dims)
{
    if (input_dims->c <= 0 || input_dims->c >= ARM_NN_CONV_SMALL_C_F16_LANES || output_dims->c <= 0 ||
        output_dims->w <= 0)
    {
        return false;
    }
    /* u16 gather/scatter offsets are relative to one input row / one output position group. */
    if ((size_t)input_dims->w * (size_t)input_dims->c > ARM_NN_MVE_F16_GATHER_OFFSET_MAX ||
        (size_t)ARM_NN_CONV_SMALL_C_F16_LANES * (size_t)output_dims->c > ARM_NN_MVE_F16_GATHER_OFFSET_MAX)
    {
        return false;
    }
    return true;
}

/* Weight of output channel oc at packed-patch index k (tap * input_c + ic), OHWI row or NT_N_PACKED lane.
 * Never called with oc >= output_c, so a partial last packed block is never read past its live lanes. */
__STATIC_FORCEINLINE float16_t
arm_conv_small_c_weight_f16(const float16_t *filter, bool packed, int32_t patch_len, int32_t oc, int32_t k)
{
    if (packed)
    {
        return filter[((size_t)(oc / 8) * (size_t)patch_len + (size_t)k) * 8 + (size_t)(oc % 8)];
    }
    return filter[(size_t)oc * (size_t)patch_len + (size_t)k];
}

/* N_OC (1..OC_GROUP) output channels from oc0 for one group of LANES output x positions. N_OC is a
 * compile-time constant at every call site so the unused accumulators fold away. */
__STATIC_FORCEINLINE void arm_conv_small_c_group_f16(const float16_t *__RESTRICT input_b,
                                                     int32_t input_h,
                                                     int32_t input_w,
                                                     int32_t input_c,
                                                     const float16_t *__RESTRICT filter,
                                                     bool packed,
                                                     int32_t patch_len,
                                                     int32_t kernel_h,
                                                     int32_t kernel_w,
                                                     int32_t in_y0,
                                                     int32_t dil_h,
                                                     int32_t dil_w,
                                                     uint16x8_t in_x_base,
                                                     const float16_t *__RESTRICT bias,
                                                     int32_t oc0,
                                                     const int32_t N_OC,
                                                     float16x8_t vmin,
                                                     float16x8_t vmax,
                                                     float16_t *__RESTRICT out_pos,
                                                     uint16x8_t out_offsets,
                                                     mve_pred16_t p_pos)
{
    float16x8_t vacc0 = bias ? vdupq_n_f16(bias[oc0]) : vdupq_n_f16((float16_t)0.0f);
    float16x8_t vacc1 = (N_OC > 1 && bias) ? vdupq_n_f16(bias[oc0 + 1]) : vdupq_n_f16((float16_t)0.0f);
    float16x8_t vacc2 = (N_OC > 2 && bias) ? vdupq_n_f16(bias[oc0 + 2]) : vdupq_n_f16((float16_t)0.0f);
    float16x8_t vacc3 = (N_OC > 3 && bias) ? vdupq_n_f16(bias[oc0 + 3]) : vdupq_n_f16((float16_t)0.0f);
    const uint16x8_t vinput_w = vdupq_n_u16((uint16_t)input_w);

    for (int32_t ky = 0; ky < kernel_h; ++ky)
    {
        const int32_t in_y = in_y0 + ky * dil_h;
        if (in_y < 0 || in_y >= input_h)
        {
            continue;
        }
        const float16_t *row = input_b + (size_t)in_y * (size_t)input_w * (size_t)input_c;
        for (int32_t kx = 0; kx < kernel_w; ++kx)
        {
            /* Unsigned wrap turns a negative (padded) in_x into a huge value, so one compare covers both edges. */
            const uint16x8_t in_x = vaddq(in_x_base, (uint16_t)(kx * dil_w));
            const mve_pred16_t p = vcmphiq_m(vinput_w, in_x, p_pos);
            const uint16x8_t off_x = vmulq(in_x, (uint16_t)input_c);
            const int32_t k_tap = (ky * kernel_w + kx) * input_c;
            for (int32_t ic = 0; ic < input_c; ++ic)
            {
                const float16x8_t vin = vldrhq_gather_shifted_offset_z(row, vaddq(off_x, (uint16_t)ic), p);
                const int32_t k = k_tap + ic;
                vacc0 = vfmaq(vacc0, vin, arm_conv_small_c_weight_f16(filter, packed, patch_len, oc0, k));
                if (N_OC > 1)
                {
                    vacc1 = vfmaq(vacc1, vin, arm_conv_small_c_weight_f16(filter, packed, patch_len, oc0 + 1, k));
                }
                if (N_OC > 2)
                {
                    vacc2 = vfmaq(vacc2, vin, arm_conv_small_c_weight_f16(filter, packed, patch_len, oc0 + 2, k));
                }
                if (N_OC > 3)
                {
                    vacc3 = vfmaq(vacc3, vin, arm_conv_small_c_weight_f16(filter, packed, patch_len, oc0 + 3, k));
                }
            }
        }
    }

    vstrhq_scatter_shifted_offset_p(out_pos + oc0, out_offsets, arm_nn_clamp_mve_f16(vacc0, vmin, vmax), p_pos);
    if (N_OC > 1)
    {
        vstrhq_scatter_shifted_offset_p(out_pos + oc0 + 1, out_offsets, arm_nn_clamp_mve_f16(vacc1, vmin, vmax), p_pos);
    }
    if (N_OC > 2)
    {
        vstrhq_scatter_shifted_offset_p(out_pos + oc0 + 2, out_offsets, arm_nn_clamp_mve_f16(vacc2, vmin, vmax), p_pos);
    }
    if (N_OC > 3)
    {
        vstrhq_scatter_shifted_offset_p(out_pos + oc0 + 3, out_offsets, arm_nn_clamp_mve_f16(vacc3, vmin, vmax), p_pos);
    }
}

static arm_cmsis_nn_status arm_nn_conv_small_c_nhwc_f16(const cmsis_nn_conv_params_f16 *conv_params,
                                                        const cmsis_nn_dims *input_dims,
                                                        const float16_t *input_data,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const float16_t *filter_data,
                                                        const float16_t *bias_data,
                                                        const cmsis_nn_dims *output_dims,
                                                        float16_t *output_data)
{
    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dil_h = conv_params->dilation.h;
    const int32_t dil_w = conv_params->dilation.w;
    const int32_t patch_len = kernel_h * kernel_w * input_c;
    const bool packed = conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
    const float16x8_t vmin = vdupq_n_f16(conv_params->activation.min);
    const float16x8_t vmax = vdupq_n_f16(conv_params->activation.max);
    const uint16x8_t lane = vidupq_u16(0u, 1);
    const uint16x8_t lane_x = vmulq(lane, (uint16_t)stride_w);
    const uint16x8_t out_offsets = vmulq(lane, (uint16_t)output_c);
    /* All lanes on; the only vctp is the one tail predicate per row, outside any counted loop. */
    const mve_pred16_t p_all = (mve_pred16_t)0xFFFFU;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int32_t in_y0 = out_y * stride_h - pad_h;
            float16_t *out_row = output_b + (size_t)out_y * output_w * output_c;
            int32_t out_x0 = 0;

            for (;; out_x0 += ARM_NN_CONV_SMALL_C_F16_LANES)
            {
                const int32_t remaining = output_w - out_x0;
                const bool full = remaining >= ARM_NN_CONV_SMALL_C_F16_LANES;
                if (remaining <= 0)
                {
                    break;
                }
                const mve_pred16_t p_pos = full ? p_all : vctp16q((uint32_t)remaining);
                const uint16x8_t in_x_base = vaddq(lane_x, (uint16_t)(out_x0 * stride_w - pad_w));
                float16_t *out_pos = out_row + (size_t)out_x0 * output_c;

                for (int32_t oc0 = 0; oc0 < output_c; oc0 += ARM_NN_CONV_SMALL_C_F16_OC_GROUP)
                {
                    const int32_t n_oc = output_c - oc0;
        #define ARM_NN_CONV_SMALL_C_CALL(N)                                                                            \
            arm_conv_small_c_group_f16(input_b,                                                                        \
                                       input_h,                                                                        \
                                       input_w,                                                                        \
                                       input_c,                                                                        \
                                       filter_data,                                                                    \
                                       packed,                                                                         \
                                       patch_len,                                                                      \
                                       kernel_h,                                                                       \
                                       kernel_w,                                                                       \
                                       in_y0,                                                                          \
                                       dil_h,                                                                          \
                                       dil_w,                                                                          \
                                       in_x_base,                                                                      \
                                       bias_data,                                                                      \
                                       oc0,                                                                            \
                                       (N),                                                                            \
                                       vmin,                                                                           \
                                       vmax,                                                                           \
                                       out_pos,                                                                        \
                                       out_offsets,                                                                    \
                                       p_pos)
                    switch (n_oc)
                    {
                    case 1:
                        ARM_NN_CONV_SMALL_C_CALL(1);
                        break;
                    case 2:
                        ARM_NN_CONV_SMALL_C_CALL(2);
                        break;
                    case 3:
                        ARM_NN_CONV_SMALL_C_CALL(3);
                        break;
                    default:
                        ARM_NN_CONV_SMALL_C_CALL(4);
                        break;
                    }
        #undef ARM_NN_CONV_SMALL_C_CALL
                }
                if (!full)
                {
                    break;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
    #endif /* defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE) */

static arm_cmsis_nn_status arm_convolve_nhwc_patch_gemm_f16(const cmsis_nn_context *ctx,
                                                            const cmsis_nn_conv_params_f16 *conv_params,
                                                            const cmsis_nn_dims *input_dims,
                                                            const float16_t *input_data,
                                                            const cmsis_nn_dims *filter_dims,
                                                            const float16_t *filter_data,
                                                            const float16_t *bias_data,
                                                            const cmsis_nn_dims *output_dims,
                                                            float16_t *output_data)
{
    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dil_h = conv_params->dilation.h;
    const int32_t dil_w = conv_params->dilation.w;
    const int32_t patch_len = kernel_h * kernel_w * input_c;
    const int32_t output_positions = output_h * output_w;

    const size_t row_bytes = (size_t)patch_len * sizeof(float16_t);
    if (row_bytes == 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t tile_rows = ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MAX_TILE_ROWS;
    const int32_t max_rows_by_ctx = (int32_t)((size_t)ctx->size / row_bytes);
    if (max_rows_by_ctx < tile_rows)
    {
        tile_rows = max_rows_by_ctx;
    }
    if (tile_rows <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Developers familiar with im2row terminology: each output position becomes one packed row here. */
    float16_t *patch_matrix = (float16_t *)ctx->buf;
    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t pos = 0; pos < output_positions; pos += tile_rows)
        {
            const int32_t rows = ((output_positions - pos) < tile_rows) ? (output_positions - pos) : tile_rows;
            for (int32_t r = 0; r < rows; ++r)
            {
                const int32_t out_pos = pos + r;
                const int32_t out_y = out_pos / output_w;
                const int32_t out_x = out_pos - out_y * output_w;
                float16_t *patch_row = patch_matrix + (size_t)r * patch_len;
                arm_nn_pack_conv_patch_f16(input_b,
                                           input_h,
                                           input_w,
                                           input_c,
                                           kernel_h,
                                           kernel_w,
                                           stride_h,
                                           stride_w,
                                           pad_h,
                                           pad_w,
                                           dil_h,
                                           dil_w,
                                           out_y,
                                           out_x,
                                           (float16_t)0.0f,
                                           patch_row);
            }

            arm_cmsis_nn_status st = arm_convolve_patch_mat_mul_f16(patch_matrix,
                                                                    filter_data,
                                                                    bias_data,
                                                                    output_b + (size_t)pos * output_c,
                                                                    rows,
                                                                    output_c,
                                                                    patch_len,
                                                                    output_c,
                                                                    conv_params);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

    #include <stdio.h>

arm_cmsis_nn_status arm_convolve_nhwc_f16(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f16 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float16_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float16_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float16_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float16_t *output_data)
{
    (void)bias_dims;

    if (!conv_params || !input_dims || !filter_dims || !output_dims || !input_data || !filter_data || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dil_h = conv_params->dilation.h;
    const int32_t dil_w = conv_params->dilation.w;
    const int32_t patch_len = kernel_h * kernel_w * input_c;
    const int32_t output_positions = output_h * output_w;

    if (arm_conv_nhwc_use_1x1_f16(conv_params, filter_dims))
    {
        return arm_convolve_1x1_nhwc_f16(ctx,
                                         conv_params,
                                         input_dims,
                                         input_data,
                                         filter_dims,
                                         filter_data,
                                         bias_dims,
                                         bias_data,
                                         output_dims,
                                         output_data);
    }

    if (arm_conv_nhwc_use_1xn_f16(ctx, conv_params, input_dims, filter_dims, output_dims))
    {
        return arm_convolve_1_x_n_nhwc_f16(ctx,
                                           conv_params,
                                           input_dims,
                                           input_data,
                                           filter_dims,
                                           filter_data,
                                           bias_dims,
                                           bias_data,
                                           output_dims,
                                           output_data);
    }

    #ifndef NN_DISABLE_SPECIALIZATION
    /*
     * Let direct specializations claim their shapes first. Packed-patch GEMM
     * remains the generic fallback for shapes that are not handled by a tuned
     * direct kernel.
     */
    ARM_CONV_DISPATCH(arm_conv_spec_nhwc_f16,
                      ARM_CONV_ARRAY_SIZE(arm_conv_spec_nhwc_f16),
                      ctx,
                      conv_params,
                      input_dims,
                      input_data,
                      filter_dims,
                      filter_data,
                      bias_dims,
                      bias_data,
                      output_dims,
                      output_data);
    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (arm_conv_nhwc_use_small_c_f16(input_dims, output_dims))
    {
        return arm_nn_conv_small_c_nhwc_f16(
            conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
    }
    #endif

    const bool use_patch_gemm = arm_conv_nhwc_use_patch_gemm_f16(ctx, patch_len, output_c, output_positions);

    if (use_patch_gemm)
    {
        arm_cmsis_nn_status st = arm_convolve_nhwc_patch_gemm_f16(
            ctx, conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
        if (st == ARM_CMSIS_NN_SUCCESS)
        {
            return st;
        }
    }

    /* NT_N_PACKED filters are stored as [out_c / 8][patch][8 lanes]; index them lane-wise below rather than
     * as OHWI rows. */
    const bool weights_packed = conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vmin = vdupq_n_f16(conv_params->activation.min);
    const float16x8_t vmax = vdupq_n_f16(conv_params->activation.max);
    #endif

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int32_t in_y0 = out_y * stride_h - pad_h;
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                const int32_t in_x0 = out_x * stride_w - pad_w;
                float16_t *out_pos = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                if (weights_packed)
                {
                    /* Lanes are output channels: one weight vector per (tap, ic), no reduction (#417). The
                     * last block is loaded under the same predicate as it is stored. */
                    for (int32_t oc0 = 0; oc0 < output_c; oc0 += 8)
                    {
                        const mve_pred16_t p_oc = vctp16q((uint32_t)(output_c - oc0));
                        float16x8_t vacc = bias_data ? vld1q_z(bias_data + oc0, p_oc) : vdupq_n_f16((float16_t)0.0f);
                        const float16_t *w_blk = filter_data + (size_t)oc0 * patch_len;

                        for (int32_t ky = 0; ky < kernel_h; ++ky)
                        {
                            const int32_t in_y = in_y0 + ky * dil_h;
                            if (in_y < 0 || in_y >= input_h)
                            {
                                continue;
                            }
                            for (int32_t kx = 0; kx < kernel_w; ++kx)
                            {
                                const int32_t in_x = in_x0 + kx * dil_w;
                                if (in_x < 0 || in_x >= input_w)
                                {
                                    continue;
                                }
                                const size_t k0 = ((size_t)ky * kernel_w + (size_t)kx) * input_c;
                                const float16_t *x = input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c;
                                const float16_t *w_tap = w_blk + k0 * 8;
                                for (int32_t ic = 0; ic < input_c; ++ic)
                                {
                                    vacc = vfmaq(vacc, vld1q_z(w_tap + (size_t)ic * 8, p_oc), x[ic]);
                                }
                            }
                        }

                        vacc = arm_nn_clamp_mve_f16(vacc, vmin, vmax);
                        vst1q_p(out_pos + oc0, vacc, p_oc);
                    }
                    continue;
                }
    #endif

                for (int32_t oc = 0; oc < output_c; ++oc)
                {
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                    /* Packed weights never reach here under MVE; OHWI rows only. One accumulator is carried
                     * across every tap and reduced once per output (#417). */
                    const float16_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * input_c;
                    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
    #else
                    _Float16 acc = bias_data ? (_Float16)bias_data[oc] : (_Float16)0;
                    const float16_t *w_oc = weights_packed
                        ? filter_data + ((size_t)(oc / 8) * patch_len) * 8 + (size_t)(oc % 8)
                        : filter_data + (size_t)oc * kernel_h * kernel_w * input_c;
    #endif

                    for (int32_t ky = 0; ky < kernel_h; ++ky)
                    {
                        const int32_t in_y = in_y0 + ky * dil_h;
                        if (in_y < 0 || in_y >= input_h)
                        {
                            continue;
                        }
                        for (int32_t kx = 0; kx < kernel_w; ++kx)
                        {
                            const int32_t in_x = in_x0 + kx * dil_w;
                            if (in_x < 0 || in_x >= input_w)
                            {
                                continue;
                            }
                            const size_t k0 = ((size_t)ky * kernel_w + (size_t)kx) * input_c;
                            const float16_t *x = input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                            /* Full blocks unpredicated, one predicated tail: no vctp inside a loop. */
                            const float16_t *w_tap = w_oc + k0;
                            int32_t ic = 0;
                            for (; ic + 8 <= input_c; ic += 8)
                            {
                                vacc = vfmaq(vacc, vld1q(x + ic), vld1q(w_tap + ic));
                            }
                            if (ic < input_c)
                            {
                                const mve_pred16_t p = vctp16q((uint32_t)(input_c - ic));
                                vacc = vfmaq_m(vacc, vld1q_z(x + ic, p), vld1q_z(w_tap + ic, p), p);
                            }
    #else
                            if (weights_packed)
                            {
                                for (int32_t ic = 0; ic < input_c; ++ic)
                                {
                                    acc += (_Float16)x[ic] * (_Float16)w_oc[(k0 + (size_t)ic) * 8];
                                }
                            }
                            else
                            {
                                for (int32_t ic = 0; ic < input_c; ++ic)
                                {
                                    acc += (_Float16)x[ic] * (_Float16)w_oc[k0 + (size_t)ic];
                                }
                            }
    #endif
                        }
                    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                    _Float16 acc = bias_data ? (_Float16)bias_data[oc] : (_Float16)0;
                    acc += (_Float16)arm_nn_vec_reduce_add_f16(vacc);
    #endif
                    acc = arm_nn_clamp_f16h(
                        acc, (_Float16)conv_params->activation.max, (_Float16)conv_params->activation.min);
                    out_pos[oc] = (float16_t)acc;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_convolve_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f16 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float16_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float16_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *output_data,
                                     arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_convolve_nhwc_f16(ctx,
                                 conv_params,
                                 input_dims,
                                 input_data,
                                 filter_dims,
                                 filter_data,
                                 bias_dims,
                                 bias_data,
                                 output_dims,
                                 output_data);
}

arm_cmsis_nn_status arm_convolve_wrapper_f16(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f16 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float16_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float16_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float16_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float16_t *output_data)
{
    return arm_convolve_f16(ctx,
                            conv_params,
                            input_dims,
                            input_data,
                            filter_dims,
                            filter_data,
                            bias_dims,
                            bias_data,
                            output_dims,
                            output_data,
                            ARM_NN_LAYOUT_NHWC);
}
/**
 * @} end of NNConv group
 */

#endif /* ARM_NN_ENABLE_F16 */
