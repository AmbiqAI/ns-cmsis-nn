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
 * Title:        arm_convolve_f32.c
 * Description:  Generic float32 convolution
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

/* Generic float32 convolution. */

#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F32

    #include "Internal/arm_conv_opt_common.h"
    #include "Internal/arm_conv_opt_f32.h"
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

__STATIC_INLINE bool arm_conv_nhwc_use_patch_gemm_f32(const cmsis_nn_context *ctx,
                                                      int32_t patch_len,
                                                      int32_t output_c,
                                                      int32_t output_positions)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || patch_len <= 0)
    {
        return false;
    }

    /* No patch-length floor: the sizer already covers every patch length (#417). */
    if (output_c < ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_OC || output_positions < ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_POS)
    {
        return false;
    }

    const size_t row_bytes = (size_t)patch_len * sizeof(float32_t);
    return row_bytes > 0U && (size_t)ctx->size >= row_bytes;
}

__STATIC_INLINE bool arm_conv_nhwc_use_1x1_f32(const cmsis_nn_conv_params_f32 *conv_params,
                                               const cmsis_nn_dims *filter_dims)
{
    return conv_params && filter_dims && filter_dims->h == 1 && filter_dims->w == 1 && conv_params->padding.h == 0 &&
        conv_params->padding.w == 0;
}

__STATIC_INLINE arm_cmsis_nn_status arm_convolve_patch_mat_mul_f32(const float32_t *lhs,
                                                                   const float32_t *rhs,
                                                                   const float32_t *bias,
                                                                   float32_t *dst,
                                                                   int32_t lhs_rows,
                                                                   int32_t rhs_rows,
                                                                   int32_t rhs_cols,
                                                                   int32_t row_address_offset,
                                                                   const cmsis_nn_conv_params_f32 *conv_params)
{
    if (conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        return arm_nn_mat_mult_nt_n_packed_f32(lhs,
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

    return arm_nn_mat_mult_nt_t_f32(lhs,
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

__STATIC_INLINE bool arm_conv_nhwc_use_1xn_f32(const cmsis_nn_context *ctx,
                                               const cmsis_nn_conv_params_f32 *conv_params,
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
    if (arm_conv_spec_nhwc_f32_matches_any(ctx, conv_params, input_dims, filter_dims, output_dims))
    {
        return false;
    }
    #endif

    /* Remaining 1xN shapes use the generic packed-input helper when workspace is available. */
    const int32_t buf_size =
        arm_convolve_1_x_n_f32_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
    return buf_size > 0 && ctx->size >= buf_size;
}

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV_SMALL_C_F32_LANES (4)
        #define ARM_NN_CONV_SMALL_C_F32_OC_GROUP (4)
        #define ARM_NN_CONV_SMALL_C_MODE_EDGE (0)
        #define ARM_NN_CONV_SMALL_C_MODE_GATHER (1)
        #define ARM_NN_CONV_SMALL_C_MODE_VLD1 (2)
        #define ARM_NN_CONV_SMALL_C_MODE_VLD2 (3)
        #define ARM_NN_CONV_SMALL_C_MODE_VLD4 (4)

/* Per-call constants of the direct small-C kernel, computed once by the dispatcher (#417). */
typedef struct
{
    const float32_t *filter;
    const float32_t *bias;
    int32_t input_h;
    int32_t input_w;
    int32_t input_c;
    int32_t output_h;
    int32_t output_w;
    int32_t output_c;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dil_h;
    int32_t dil_w;
    int32_t patch_len;
    int32_t w_lane_stride; /* weight step per patch index: 1 for OHWI rows, LANES for NT_N_PACKED blocks */
    int32_t w_oc_step;     /* weight step per output channel: patch_len for OHWI rows, 1 for NT_N_PACKED */
    bool packed;
    float32x4_t vmin;
    float32x4_t vmax;
    uint32x4_t lane_x_c;    /* lane * stride_w * input_c */
    uint32x4_t out_offsets; /* lane * output_c */
} arm_conv_small_c_f32;

/* Direct kernel for fewer input channels than one vector: lanes are output x positions, weights are scalars,
 * OC_GROUP output channels accumulate at once. Interior lane groups
 * (every tap in range) load unpredicated -- contiguous for input_c == 1 with stride 1/2/4, gathered otherwise --
 * and store unpredicated; edge and tail groups compare in_x against input_w once per tap column. */
__STATIC_INLINE bool arm_conv_nhwc_use_small_c_f32(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *output_dims)
{
    if (input_dims->c <= 0 || input_dims->c >= ARM_NN_CONV_SMALL_C_F32_LANES || output_dims->c <= 0 ||
        output_dims->w <= 0)
    {
        return false;
    }
    return true;
}

/* n_oc (1..OC_GROUP) output channels from oc0 for one group of LANES output x positions whose first tap
 * column is in_x0. IC1 and MODE are compile-time constants at every call site. A partial group aliases the
 * missing channels onto oc0 (weights and bias are read for live channels only) and skips their stores, so a
 * partial last packed block is never read past its live lanes. */
__STATIC_FORCEINLINE void arm_conv_small_c_group_f32(const arm_conv_small_c_f32 *__RESTRICT c,
                                                     const float32_t *__RESTRICT input_b,
                                                     int32_t in_y0,
                                                     int32_t in_x0,
                                                     float32_t *__RESTRICT out_pos,
                                                     int32_t oc0,
                                                     int32_t n_oc,
                                                     const bool IC1,
                                                     const int32_t MODE,
                                                     mve_pred16_t p_pos)
{
    const int32_t oc_idx0 = (0 < n_oc) ? 0 : 0;
    const int32_t oc_idx1 = (1 < n_oc) ? 1 : 0;
    const int32_t oc_idx2 = (2 < n_oc) ? 2 : 0;
    const int32_t oc_idx3 = (3 < n_oc) ? 3 : 0;
    float32x4_t vacc0 = c->bias ? vdupq_n_f32(c->bias[oc0 + oc_idx0]) : vdupq_n_f32(0.0f);
    float32x4_t vacc1 = c->bias ? vdupq_n_f32(c->bias[oc0 + oc_idx1]) : vdupq_n_f32(0.0f);
    float32x4_t vacc2 = c->bias ? vdupq_n_f32(c->bias[oc0 + oc_idx2]) : vdupq_n_f32(0.0f);
    float32x4_t vacc3 = c->bias ? vdupq_n_f32(c->bias[oc0 + oc_idx3]) : vdupq_n_f32(0.0f);
    const float32_t *w0 = c->packed ? c->filter + ((size_t)(oc0 / 4) * (size_t)c->patch_len) * 4 + (size_t)(oc0 % 4)
                                    : c->filter + (size_t)oc0 * (size_t)c->patch_len;
    const int32_t ws = c->w_lane_stride;
    const int32_t w_off0 = oc_idx0 * c->w_oc_step;
    const int32_t w_off1 = oc_idx1 * c->w_oc_step;
    const int32_t w_off2 = oc_idx2 * c->w_oc_step;
    const int32_t w_off3 = oc_idx3 * c->w_oc_step;
    const int32_t input_c = c->input_c;
    /* Element offset of the first input column past the row; in_x wraps negative to a huge unsigned value
     * and the routing check keeps every reachable in_x * input_c inside the offset type. */
    const uint32_t row_limit = (uint32_t)(c->input_w * input_c);
    const int32_t kernel_w = c->kernel_w;
    const int32_t dil_w = c->dil_w;
    const size_t row_len = (size_t)c->input_w * (size_t)input_c;

    for (int32_t ky = 0; ky < c->kernel_h; ++ky)
    {
        const int32_t in_y = in_y0 + ky * c->dil_h;
        if (in_y < 0 || in_y >= c->input_h)
        {
            continue;
        }
        const float32_t *row = input_b + (size_t)in_y * row_len;
        for (int32_t kx = 0; kx < kernel_w; ++kx)
        {
            const int32_t x0 = in_x0 + kx * dil_w;
            const float32_t *wt = w0 + (size_t)((ky * kernel_w + kx) * input_c) * (size_t)ws;
            if (IC1)
            {
                float32x4_t vin;
                if (MODE == ARM_NN_CONV_SMALL_C_MODE_VLD1)
                {
                    vin = vld1q(row + x0);
                }
                else if (MODE == ARM_NN_CONV_SMALL_C_MODE_VLD4)
                {
                    vin = vld4q(row + x0).val[0];
                }
                else
                {
                    vin = vld2q(row + x0).val[0];
                }
                vacc0 = vfmaq(vacc0, vin, wt[w_off0]);
                vacc1 = vfmaq(vacc1, vin, wt[w_off1]);
                vacc2 = vfmaq(vacc2, vin, wt[w_off2]);
                vacc3 = vfmaq(vacc3, vin, wt[w_off3]);
            }
            else
            {
                /* Re-read the lane offsets from the context each tap column rather than holding them in a Q
                 * register across the channel loop; edge groups compare them against the row limit once per
                 * column, for every channel. */
                __asm__ volatile("" : "+r"(c));
                const uint32x4_t off = vaddq(c->lane_x_c, (uint32_t)(x0 * input_c));
                const mve_pred16_t p =
                    (MODE == ARM_NN_CONV_SMALL_C_MODE_EDGE) ? vcmphiq_m(vdupq_n_u32(row_limit), off, p_pos) : p_pos;
                for (int32_t ic = 0; ic < input_c; ++ic)
                {
                    const float32x4_t vin = (MODE == ARM_NN_CONV_SMALL_C_MODE_EDGE)
                        ? vldrwq_gather_shifted_offset_z(row + ic, off, p)
                        : vldrwq_gather_shifted_offset(row + ic, off);
                    vacc0 = vfmaq(vacc0, vin, wt[w_off0]);
                    vacc1 = vfmaq(vacc1, vin, wt[w_off1]);
                    vacc2 = vfmaq(vacc2, vin, wt[w_off2]);
                    vacc3 = vfmaq(vacc3, vin, wt[w_off3]);
                    wt += ws;
                }
            }
        }
    }

    const float32x4_t vmin = c->vmin;
    const float32x4_t vmax = c->vmax;
    const uint32x4_t out_offsets = c->out_offsets;
    if (MODE == ARM_NN_CONV_SMALL_C_MODE_EDGE)
    {
        vstrwq_scatter_shifted_offset_p(out_pos + oc0, out_offsets, arm_nn_clamp_mve_f32(vacc0, vmin, vmax), p_pos);
    }
    else
    {
        vstrwq_scatter_shifted_offset(out_pos + oc0, out_offsets, arm_nn_clamp_mve_f32(vacc0, vmin, vmax));
    }
    if (n_oc > 1)
    {
        if (MODE == ARM_NN_CONV_SMALL_C_MODE_EDGE)
        {
            vstrwq_scatter_shifted_offset_p(
                out_pos + oc0 + 1, out_offsets, arm_nn_clamp_mve_f32(vacc1, vmin, vmax), p_pos);
        }
        else
        {
            vstrwq_scatter_shifted_offset(out_pos + oc0 + 1, out_offsets, arm_nn_clamp_mve_f32(vacc1, vmin, vmax));
        }
    }
    if (n_oc > 2)
    {
        if (MODE == ARM_NN_CONV_SMALL_C_MODE_EDGE)
        {
            vstrwq_scatter_shifted_offset_p(
                out_pos + oc0 + 2, out_offsets, arm_nn_clamp_mve_f32(vacc2, vmin, vmax), p_pos);
        }
        else
        {
            vstrwq_scatter_shifted_offset(out_pos + oc0 + 2, out_offsets, arm_nn_clamp_mve_f32(vacc2, vmin, vmax));
        }
    }
    if (n_oc > 3)
    {
        if (MODE == ARM_NN_CONV_SMALL_C_MODE_EDGE)
        {
            vstrwq_scatter_shifted_offset_p(
                out_pos + oc0 + 3, out_offsets, arm_nn_clamp_mve_f32(vacc3, vmin, vmax), p_pos);
        }
        else
        {
            vstrwq_scatter_shifted_offset(out_pos + oc0 + 3, out_offsets, arm_nn_clamp_mve_f32(vacc3, vmin, vmax));
        }
    }
}

/* Every output channel of one lane group, OC_GROUP at a time. */
__STATIC_FORCEINLINE void arm_conv_small_c_row_group_f32(const arm_conv_small_c_f32 *__RESTRICT c,
                                                         const float32_t *__RESTRICT input_b,
                                                         int32_t in_y0,
                                                         int32_t in_x0,
                                                         float32_t *__RESTRICT out_pos,
                                                         const bool IC1,
                                                         const int32_t MODE,
                                                         mve_pred16_t p_pos)
{
    for (int32_t oc0 = 0; oc0 < c->output_c; oc0 += ARM_NN_CONV_SMALL_C_F32_OC_GROUP)
    {
        const int32_t n_oc = c->output_c - oc0;
        arm_conv_small_c_group_f32(c,
                                   input_b,
                                   in_y0,
                                   in_x0,
                                   out_pos,
                                   oc0,
                                   (n_oc < ARM_NN_CONV_SMALL_C_F32_OC_GROUP) ? n_oc : ARM_NN_CONV_SMALL_C_F32_OC_GROUP,
                                   IC1,
                                   MODE,
                                   p_pos);
    }
}

/* One lane group; MODE is resolved here so each instantiation above sees a constant. input_c == 1 with a
 * stride that has no contiguous load form takes the generic path with a one-trip channel loop. */
__STATIC_FORCEINLINE void arm_conv_small_c_lane_group_f32(const arm_conv_small_c_f32 *__RESTRICT c,
                                                          const float32_t *__RESTRICT input_b,
                                                          int32_t in_y0,
                                                          int32_t in_x0,
                                                          float32_t *__RESTRICT out_pos,
                                                          int32_t mode,
                                                          mve_pred16_t p_pos)
{
    switch (mode)
    {
    case ARM_NN_CONV_SMALL_C_MODE_VLD1:
        arm_conv_small_c_row_group_f32(c, input_b, in_y0, in_x0, out_pos, true, ARM_NN_CONV_SMALL_C_MODE_VLD1, p_pos);
        break;
    case ARM_NN_CONV_SMALL_C_MODE_VLD2:
        arm_conv_small_c_row_group_f32(c, input_b, in_y0, in_x0, out_pos, true, ARM_NN_CONV_SMALL_C_MODE_VLD2, p_pos);
        break;
    case ARM_NN_CONV_SMALL_C_MODE_VLD4:
        arm_conv_small_c_row_group_f32(c, input_b, in_y0, in_x0, out_pos, true, ARM_NN_CONV_SMALL_C_MODE_VLD4, p_pos);
        break;
    case ARM_NN_CONV_SMALL_C_MODE_GATHER:
        arm_conv_small_c_row_group_f32(
            c, input_b, in_y0, in_x0, out_pos, false, ARM_NN_CONV_SMALL_C_MODE_GATHER, p_pos);
        break;
    default:
        arm_conv_small_c_row_group_f32(c, input_b, in_y0, in_x0, out_pos, false, ARM_NN_CONV_SMALL_C_MODE_EDGE, p_pos);
        break;
    }
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_nn_conv_small_c_nhwc_f32(const arm_conv_small_c_f32 *__RESTRICT c,
                             int32_t batch,
                             const float32_t *__RESTRICT input_data,
                             float32_t *__RESTRICT output_data)
{
    /* Keep the context behind a pointer: without this GCC's IPA-SRA clones the kernel, scalarises the
     * struct, and reloads the lane vectors from its own stack copy inside the tap loops (#417). */
    __asm__ volatile("" : "+r"(c));
    const int32_t input_w = c->input_w;
    const int32_t output_w = c->output_w;
    const int32_t output_c = c->output_c;
    const int32_t stride_w = c->stride_w;
    /* Span of input columns one full lane group touches, minus one. */
    const int32_t span = (ARM_NN_CONV_SMALL_C_F32_LANES - 1) * stride_w + (c->kernel_w - 1) * c->dil_w;
    int32_t interior_mode = ARM_NN_CONV_SMALL_C_MODE_GATHER;
    if (c->input_c == 1)
    {
        if (stride_w == 1)
        {
            interior_mode = ARM_NN_CONV_SMALL_C_MODE_VLD1;
        }
        else if (stride_w == 2)
        {
            interior_mode = ARM_NN_CONV_SMALL_C_MODE_VLD2;
        }
        else if (stride_w == 4)
        {
            interior_mode = ARM_NN_CONV_SMALL_C_MODE_VLD4;
        }
    }
    /* All lanes on for full groups; the only vctp is the straight-line tail below. */
    const mve_pred16_t p_all = (mve_pred16_t)0xFFFFU;
    const size_t input_b_len = (size_t)c->input_h * (size_t)input_w * (size_t)c->input_c;
    const size_t output_b_len = (size_t)c->output_h * (size_t)output_w * (size_t)output_c;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float32_t *input_b = input_data + (size_t)b * input_b_len;
        float32_t *output_b = output_data + (size_t)b * output_b_len;

        for (int32_t out_y = 0; out_y < c->output_h; ++out_y)
        {
            const int32_t in_y0 = out_y * c->stride_h - c->pad_h;
            float32_t *out_row = output_b + (size_t)out_y * output_w * output_c;
            int32_t out_x0 = 0;

            for (; out_x0 + ARM_NN_CONV_SMALL_C_F32_LANES <= output_w; out_x0 += ARM_NN_CONV_SMALL_C_F32_LANES)
            {
                const int32_t in_x0 = out_x0 * stride_w - c->pad_w;
                const bool interior = in_x0 >= 0 && in_x0 + span < input_w;
                arm_conv_small_c_lane_group_f32(c,
                                                input_b,
                                                in_y0,
                                                in_x0,
                                                out_row + (size_t)out_x0 * output_c,
                                                interior ? interior_mode : ARM_NN_CONV_SMALL_C_MODE_EDGE,
                                                p_all);
            }
            if (out_x0 < output_w)
            {
                const mve_pred16_t p_tail = vctp32q((uint32_t)(output_w - out_x0));
                arm_conv_small_c_lane_group_f32(c,
                                                input_b,
                                                in_y0,
                                                out_x0 * stride_w - c->pad_w,
                                                out_row + (size_t)out_x0 * output_c,
                                                ARM_NN_CONV_SMALL_C_MODE_EDGE,
                                                p_tail);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_conv_small_c_dispatch_f32(const cmsis_nn_conv_params_f32 *conv_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const float32_t *input_data,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const float32_t *filter_data,
                                                         const float32_t *bias_data,
                                                         const cmsis_nn_dims *output_dims,
                                                         float32_t *output_data)
{
    arm_conv_small_c_f32 c;
    const uint32x4_t lane = vidupq_u32(0u, 1);

    c.filter = filter_data;
    c.bias = bias_data;
    c.input_h = input_dims->h;
    c.input_w = input_dims->w;
    c.input_c = input_dims->c;
    c.output_h = output_dims->h;
    c.output_w = output_dims->w;
    c.output_c = output_dims->c;
    c.kernel_h = filter_dims->h;
    c.kernel_w = filter_dims->w;
    c.stride_h = conv_params->stride.h;
    c.stride_w = conv_params->stride.w;
    c.pad_h = conv_params->padding.h;
    c.pad_w = conv_params->padding.w;
    c.dil_h = conv_params->dilation.h;
    c.dil_w = conv_params->dilation.w;
    c.patch_len = c.kernel_h * c.kernel_w * c.input_c;
    c.packed = conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
    c.w_lane_stride = c.packed ? ARM_NN_CONV_SMALL_C_F32_LANES : 1;
    c.w_oc_step = c.packed ? 1 : c.patch_len;
    c.vmin = vdupq_n_f32(conv_params->activation.min);
    c.vmax = vdupq_n_f32(conv_params->activation.max);
    c.lane_x_c = vmulq(lane, (uint32_t)(c.stride_w * c.input_c));
    c.out_offsets = vmulq(lane, (uint32_t)c.output_c);

    return arm_nn_conv_small_c_nhwc_f32(&c, input_dims->n, input_data, output_data);
}
    #endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */

static arm_cmsis_nn_status arm_convolve_nhwc_patch_gemm_f32(const cmsis_nn_context *ctx,
                                                            const cmsis_nn_conv_params_f32 *conv_params,
                                                            const cmsis_nn_dims *input_dims,
                                                            const float32_t *input_data,
                                                            const cmsis_nn_dims *filter_dims,
                                                            const float32_t *filter_data,
                                                            const float32_t *bias_data,
                                                            const cmsis_nn_dims *output_dims,
                                                            float32_t *output_data)
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

    const size_t row_bytes = (size_t)patch_len * sizeof(float32_t);
    if (row_bytes == 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t tile_rows = ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MAX_TILE_ROWS;
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
    float32_t *patch_matrix = (float32_t *)ctx->buf;
    for (int32_t b = 0; b < batch; ++b)
    {
        const float32_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float32_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t pos = 0; pos < output_positions; pos += tile_rows)
        {
            const int32_t rows = ((output_positions - pos) < tile_rows) ? (output_positions - pos) : tile_rows;
            for (int32_t r = 0; r < rows; ++r)
            {
                const int32_t out_pos = pos + r;
                const int32_t out_y = out_pos / output_w;
                const int32_t out_x = out_pos - out_y * output_w;
                float32_t *patch_row = patch_matrix + (size_t)r * patch_len;
                arm_nn_pack_conv_patch_f32(input_b,
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
                                           0.0f,
                                           patch_row);
            }

            arm_cmsis_nn_status st = arm_convolve_patch_mat_mul_f32(patch_matrix,
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

arm_cmsis_nn_status arm_convolve_nhwc_f32(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f32 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float32_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float32_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float32_t *output_data)
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

    if (arm_conv_nhwc_use_1x1_f32(conv_params, filter_dims))
    {
        return arm_convolve_1x1_nhwc_f32(ctx,
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

    if (arm_conv_nhwc_use_1xn_f32(ctx, conv_params, input_dims, filter_dims, output_dims))
    {
        return arm_convolve_1_x_n_nhwc_f32(ctx,
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
    ARM_CONV_DISPATCH(arm_conv_spec_nhwc_f32,
                      ARM_CONV_ARRAY_SIZE(arm_conv_spec_nhwc_f32),
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

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (arm_conv_nhwc_use_small_c_f32(input_dims, output_dims))
    {
        return arm_conv_small_c_dispatch_f32(
            conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
    }
    #endif

    const bool use_patch_gemm = arm_conv_nhwc_use_patch_gemm_f32(ctx, patch_len, output_c, output_positions);

    if (use_patch_gemm)
    {
        arm_cmsis_nn_status st = arm_convolve_nhwc_patch_gemm_f32(
            ctx, conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
        if (st == ARM_CMSIS_NN_SUCCESS)
        {
            return st;
        }
    }

    /* NT_N_PACKED filters are stored as [out_c / 4][patch][4 lanes]; index them lane-wise below rather than
     * as OHWI rows. */
    const bool weights_packed = conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vmin = vdupq_n_f32(conv_params->activation.min);
    const float32x4_t vmax = vdupq_n_f32(conv_params->activation.max);
    #endif

    for (int32_t b = 0; b < batch; ++b)
    {
        const float32_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float32_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int32_t in_y0 = out_y * stride_h - pad_h;
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                const int32_t in_x0 = out_x * stride_w - pad_w;
                float32_t *out_pos = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                if (weights_packed)
                {
                    /* Lanes are output channels: one weight vector per (tap, ic), no reduction (#417). The
                     * last block is loaded under the same predicate as it is stored. */
                    for (int32_t oc0 = 0; oc0 < output_c; oc0 += 4)
                    {
                        const mve_pred16_t p_oc = vctp32q((uint32_t)(output_c - oc0));
                        float32x4_t vacc = bias_data ? vld1q_z(bias_data + oc0, p_oc) : vdupq_n_f32(0.0f);
                        const float32_t *w_blk = filter_data + (size_t)oc0 * patch_len;

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
                                const float32_t *x = input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c;
                                const float32_t *w_tap = w_blk + k0 * 4;
                                for (int32_t ic = 0; ic < input_c; ++ic)
                                {
                                    vacc = vfmaq(vacc, vld1q_z(w_tap + (size_t)ic * 4, p_oc), x[ic]);
                                }
                            }
                        }

                        vacc = arm_nn_clamp_mve_f32(vacc, vmin, vmax);
                        vst1q_p(out_pos + oc0, vacc, p_oc);
                    }
                    continue;
                }
    #endif

                for (int32_t oc = 0; oc < output_c; ++oc)
                {
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                    /* Packed weights never reach here under MVE; OHWI rows only. One accumulator is carried
                     * across every tap and reduced once per output (#417). */
                    const float32_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * input_c;
                    float32x4_t vacc = vdupq_n_f32(0.0f);
    #else
                    float32_t acc = bias_data ? bias_data[oc] : 0.0f;
                    const float32_t *w_oc = weights_packed
                        ? filter_data + ((size_t)(oc / 4) * patch_len) * 4 + (size_t)(oc % 4)
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
                            const float32_t *x = input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                            /* Full blocks unpredicated, one predicated tail: no vctp inside a loop. */
                            const float32_t *w_tap = w_oc + k0;
                            int32_t ic = 0;
                            for (; ic + 4 <= input_c; ic += 4)
                            {
                                vacc = vfmaq(vacc, vld1q(x + ic), vld1q(w_tap + ic));
                            }
                            if (ic < input_c)
                            {
                                const mve_pred16_t p = vctp32q((uint32_t)(input_c - ic));
                                vacc = vfmaq_m(vacc, vld1q_z(x + ic, p), vld1q_z(w_tap + ic, p), p);
                            }
    #else
                            if (weights_packed)
                            {
                                for (int32_t ic = 0; ic < input_c; ++ic)
                                {
                                    acc += x[ic] * w_oc[(k0 + (size_t)ic) * 4];
                                }
                            }
                            else
                            {
                                for (int32_t ic = 0; ic < input_c; ++ic)
                                {
                                    acc += x[ic] * w_oc[k0 + (size_t)ic];
                                }
                            }
    #endif
                        }
                    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                    float32_t acc = bias_data ? bias_data[oc] : 0.0f;
                    acc += arm_nn_vec_reduce_add_f32(vacc);
    #endif
                    acc = ARM_NN_CLAMP(acc, conv_params->activation.max, conv_params->activation.min);
                    out_pos[oc] = acc;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_convolve_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f32 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float32_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float32_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *output_data,
                                     arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_convolve_nhwc_f32(ctx,
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

arm_cmsis_nn_status arm_convolve_wrapper_f32(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f32 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float32_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float32_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float32_t *output_data)
{
    return arm_convolve_f32(ctx,
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

#endif /* ARM_NN_ENABLE_F32 */
