/*
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Copyright (c) 2026 Ambiq Micro, Inc.
 */

/* Keep grouped dispatch in its own translation unit so the compiler preserves the
 * single-group convolution loops used by existing models. */
#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F16
    #include "Internal/arm_conv_opt_f16.h"
    #include "Internal/arm_nn_activation_flt.h"
    #include "Internal/arm_nn_conv_grouped_f16.h"
    #include "arm_nnfunctions.h"
    #include "arm_nnsupportfunctions.h"

__STATIC_INLINE bool arm_convolve_nhwc_f16_grouped_args_ok(const cmsis_nn_conv_params_f16 *conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float16_t *input_data,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const float16_t *filter_data,
                                                           const cmsis_nn_dims *output_dims,
                                                           const float16_t *output_data)
{
    return conv_params && input_dims && filter_dims && output_dims && input_data && filter_data && output_data &&
        arm_nn_conv_flt_grouped_dims_ok(&conv_params->stride,
                                        &conv_params->padding,
                                        &conv_params->dilation,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        sizeof(float16_t),
                                        8,
                                        conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED);
}

__STATIC_FORCEINLINE arm_cmsis_nn_status arm_grouped_ohwi_f16(const cmsis_nn_conv_params_f16 *conv_params,
                                                              const cmsis_nn_dims *input_dims,
                                                              const float16_t *input_data,
                                                              const cmsis_nn_dims *filter_dims,
                                                              const float16_t *filter_data,
                                                              const float16_t *bias_data,
                                                              const cmsis_nn_dims *output_dims,
                                                              float16_t *output_data,
                                                              const int32_t block)
{
    (void)block; /* the scalar leg accumulates in float32 and never folds */
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
    const int32_t kernel_ch = filter_dims->c;
    const int32_t groups = input_c / kernel_ch;
    const int32_t output_ch_per_group = output_c / groups;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const bool fold = (int64_t)kernel_h * kernel_w * kernel_ch > block;
    #endif

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int64_t in_y0 = (int64_t)out_y * stride_h - pad_h;
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                const int64_t in_x0 = (int64_t)out_x * stride_w - pad_w;
                float16_t *out_pos = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;

                int32_t ky_first, ky_end, kx_first, kx_end;
                uint32_t iy_first, ix_first;
                arm_conv_f16_tap_span(in_y0, dil_h, kernel_h, input_h, &ky_first, &ky_end, &iy_first);
                arm_conv_f16_tap_span(in_x0, dil_w, kernel_w, input_w, &kx_first, &kx_end, &ix_first);
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                const bool fold_px = fold && (int64_t)(ky_end - ky_first) * (kx_end - kx_first) * kernel_ch > block;
    #else
                /* Advance integer offsets so a final dilated step never forms an out-of-range pointer. */
                const bool any_tap = ky_first < ky_end && kx_first < kx_end;
                const size_t x_first = ((size_t)iy_first * input_w + ix_first) * input_c;
                const size_t w_first = any_tap ? ((size_t)ky_first * kernel_w + (size_t)kx_first) * kernel_ch : 0;
                const size_t x_kx_step = (size_t)dil_w * input_c;
                const size_t x_ky_step = (size_t)dil_h * input_w * input_c;
                const size_t w_kx_step = (size_t)kernel_ch;
                const size_t w_ky_step = (size_t)kernel_w * kernel_ch;
    #endif

                for (int32_t group = 0; group < groups; ++group)
                {
                    const int32_t in_ch_start = group * kernel_ch;
                    const int32_t oc_end = (group + 1) * output_ch_per_group;
                    for (int32_t oc = group * output_ch_per_group; oc < oc_end; ++oc)
                    {
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        /* Carry the OHWI accumulator across taps and reduce once per output (#417). */
                        const float16_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * kernel_ch;
                        float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
                        /* Folding (#586): at most `block` channel vectors per float16 partial, each block's lanes
                         * folded into float32 pair accumulators, summed once; the bias is added in float32 and the
                         * total rounds once. */
                        float32x4_t acc_pairs = vdupq_n_f32(0.0f);
                        bool first_block = true;
                        int32_t n_vec = 0;
    #else
                        /* Scalar leg: float32 accumulation, one f16 rounding at the store (#449, #457). */
                        float32_t acc32 = bias_data ? (float32_t)bias_data[oc] : 0.0f;
                        const float16_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * kernel_ch;
                        size_t x_row = x_first + in_ch_start;
                        size_t w_row = w_first;
                        for (int32_t ky = ky_first; ky < ky_end; ++ky, x_row += x_ky_step, w_row += w_ky_step)
                        {
                            size_t x_offset = x_row;
                            size_t w_offset = w_row;
                            for (int32_t kx = kx_first; kx < kx_end; ++kx, x_offset += x_kx_step, w_offset += w_kx_step)
                            {
                                const float16_t *x = input_b + x_offset;
                                const float16_t *w = w_oc + w_offset;
                                for (int32_t ic = 0; ic < kernel_ch; ++ic)
                                {
                                    acc32 += (float32_t)x[ic] * (float32_t)w[ic];
                                }
                            }
                        }
    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        uint32_t in_y = iy_first;
                        for (int32_t ky = ky_first; ky < ky_end; ++ky, in_y += (uint32_t)dil_h)
                        {
                            uint32_t in_x = ix_first;
                            for (int32_t kx = kx_first; kx < kx_end; ++kx, in_x += (uint32_t)dil_w)
                            {
                                const size_t k0 = ((size_t)ky * kernel_w + (size_t)kx) * kernel_ch;
                                const float16_t *x =
                                    input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c + in_ch_start;
                                /* Full blocks unpredicated, one predicated tail: no vctp inside a loop. */
                                const float16_t *w_tap = w_oc + k0;
                                int32_t ic = 0;
                                while (ic < kernel_ch)
                                {
                                    int32_t ic_end = kernel_ch;
                                    if (fold_px)
                                    {
                                        if (n_vec == block)
                                        {
                                            arm_nn_f16_fold_pairs_f32(&acc_pairs, vacc, first_block);
                                            first_block = false;
                                            vacc = vdupq_n_f16((float16_t)0.0f);
                                            n_vec = 0;
                                        }
                                        const int32_t n_left = (kernel_ch - ic + 7) / 8;
                                        if (n_left > block - n_vec)
                                        {
                                            ic_end = ic + (block - n_vec) * 8;
                                        }
                                        n_vec += (ic_end - ic + 7) / 8;
                                    }
                                    for (; ic + 8 <= ic_end; ic += 8)
                                    {
                                        vacc = vfmaq(vacc, vld1q(x + ic), vld1q(w_tap + ic));
                                    }
                                    if (ic < ic_end)
                                    {
                                        const mve_pred16_t p = vctp16q((uint32_t)(ic_end - ic));
                                        vacc = vfmaq_m(vacc, vld1q_z(x + ic, p), vld1q_z(w_tap + ic, p), p);
                                        ic = ic_end;
                                    }
                                }
                            }
                        }
    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        _Float16 acc;
                        if (fold_px)
                        {
                            arm_nn_f16_fold_pairs_f32(&acc_pairs, vacc, first_block);
                            acc = (_Float16)((bias_data ? (float32_t)bias_data[oc] : 0.0f) +
                                             arm_nn_f16_pairs_sum_f32(acc_pairs));
                        }
                        else
                        {
                            acc = (_Float16)0;
                            if (bias_data)
                            {
                                acc = (_Float16)bias_data[oc];
                            }
                            acc += (_Float16)arm_nn_vec_reduce_add_f16(vacc);
                        }
    #else
                        /* Single rounding; the f16 clamp below keeps the NaN/Inf contract unchanged. */
                        _Float16 acc = (_Float16)acc32;
    #endif
                        acc = arm_nn_clamp_grouped_conv_f16(
                            acc, (_Float16)conv_params->activation.min, (_Float16)conv_params->activation.max);
                        out_pos[oc] = (float16_t)acc;
                    }
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

    #ifndef NN_DISABLE_SPECIALIZATION
static arm_cmsis_nn_status arm_try_grouped_specialization_f16(const cmsis_nn_context *ctx,
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
        #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const arm_cmsis_nn_status st = arm_convolve_f16_fast_small_kernel(ctx,
                                                                      conv_params,
                                                                      input_dims,
                                                                      input_data,
                                                                      filter_dims,
                                                                      filter_data,
                                                                      bias_dims,
                                                                      bias_data,
                                                                      output_dims,
                                                                      output_data);
    if (st != ARM_CMSIS_NN_NO_IMPL_ERROR)
    {
        return st;
    }
        #endif
    if (filter_dims->c == 1 && input_dims->c == output_dims->c)
    {
        return arm_convolve_f16_group_ch_mult_1(ctx,
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
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
}

        #define ARM_GROUPED_F16_SPECIALIZATION arm_try_grouped_specialization_f16
    #else
        #define ARM_GROUPED_F16_SPECIALIZATION NULL
    #endif

__STATIC_FORCEINLINE arm_cmsis_nn_status arm_convolve_grouped_f16(const cmsis_nn_context *ctx,
                                                                  const cmsis_nn_conv_params_f16 *conv_params,
                                                                  const cmsis_nn_dims *input_dims,
                                                                  const float16_t *input_data,
                                                                  const cmsis_nn_dims *filter_dims,
                                                                  const float16_t *filter_data,
                                                                  const cmsis_nn_dims *bias_dims,
                                                                  const float16_t *bias_data,
                                                                  const cmsis_nn_dims *output_dims,
                                                                  float16_t *output_data,
                                                                  int32_t block,
                                                                  arm_conv_call_f16 try_specialization)
{
    if (!arm_convolve_nhwc_f16_grouped_args_ok(
            conv_params, input_dims, input_data, filter_dims, filter_data, output_dims, output_data))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (conv_params->weight_format != ARM_NN_WEIGHT_FORMAT_STANDARD)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    if (arm_nn_conv_flt_empty(output_dims))
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    /* A direct OHWI caller passes NULL so section GC can drop the router's specializations. */
    if (try_specialization)
    {
        const arm_cmsis_nn_status status = try_specialization(ctx,
                                                              conv_params,
                                                              input_dims,
                                                              input_data,
                                                              filter_dims,
                                                              filter_data,
                                                              bias_dims,
                                                              bias_data,
                                                              output_dims,
                                                              output_data);
        if (status != ARM_CMSIS_NN_NO_IMPL_ERROR)
        {
            return status;
        }
    }
    return arm_grouped_ohwi_f16(
        conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data, block);
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_grouped_f16_fold(const cmsis_nn_context *ctx,
                              const cmsis_nn_conv_params_f16 *conv_params,
                              const cmsis_nn_dims *input_dims,
                              const float16_t *input_data,
                              const cmsis_nn_dims *filter_dims,
                              const float16_t *filter_data,
                              const cmsis_nn_dims *bias_dims,
                              const float16_t *bias_data,
                              const cmsis_nn_dims *output_dims,
                              float16_t *output_data,
                              arm_conv_call_f16 try_specialization)
{
    return arm_convolve_grouped_f16(ctx,
                                    conv_params,
                                    input_dims,
                                    input_data,
                                    filter_dims,
                                    filter_data,
                                    bias_dims,
                                    bias_data,
                                    output_dims,
                                    output_data,
                                    ARM_NN_F16_ACC_BLOCK,
                                    try_specialization);
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_grouped_f16_acc16(const cmsis_nn_context *ctx,
                               const cmsis_nn_conv_params_f16 *conv_params,
                               const cmsis_nn_dims *input_dims,
                               const float16_t *input_data,
                               const cmsis_nn_dims *filter_dims,
                               const float16_t *filter_data,
                               const cmsis_nn_dims *bias_dims,
                               const float16_t *bias_data,
                               const cmsis_nn_dims *output_dims,
                               float16_t *output_data,
                               arm_conv_call_f16 try_specialization)
{
    return arm_convolve_grouped_f16(ctx,
                                    conv_params,
                                    input_dims,
                                    input_data,
                                    filter_dims,
                                    filter_data,
                                    bias_dims,
                                    bias_data,
                                    output_dims,
                                    output_data,
                                    ARM_NN_F16_ACC_BLOCK_NONE,
                                    try_specialization);
}

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
    if (input_dims && filter_dims && input_dims->c != filter_dims->c)
    {
        return arm_convolve_grouped_f16_fold(ctx,
                                             conv_params,
                                             input_dims,
                                             input_data,
                                             filter_dims,
                                             filter_data,
                                             bias_dims,
                                             bias_data,
                                             output_dims,
                                             output_data,
                                             ARM_GROUPED_F16_SPECIALIZATION);
    }
    return arm_nn_convolve_single_group_nhwc_f16(ctx,
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

arm_cmsis_nn_status arm_convolve_direct_nhwc_ohwi_f16(const cmsis_nn_context *ctx,
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
    if (input_dims && filter_dims && input_dims->c != filter_dims->c)
    {
        return arm_convolve_grouped_f16_fold(ctx,
                                             conv_params,
                                             input_dims,
                                             input_data,
                                             filter_dims,
                                             filter_data,
                                             bias_dims,
                                             bias_data,
                                             output_dims,
                                             output_data,
                                             NULL);
    }
    return arm_nn_convolve_single_group_direct_nhwc_ohwi_f16(ctx,
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

arm_cmsis_nn_status arm_convolve_nhwc_f16_acc16(const cmsis_nn_context *ctx,
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
    if (input_dims && filter_dims && input_dims->c != filter_dims->c)
    {
        return arm_convolve_grouped_f16_acc16(ctx,
                                              conv_params,
                                              input_dims,
                                              input_data,
                                              filter_dims,
                                              filter_data,
                                              bias_dims,
                                              bias_data,
                                              output_dims,
                                              output_data,
                                              ARM_GROUPED_F16_SPECIALIZATION);
    }
    return arm_nn_convolve_single_group_nhwc_f16_acc16(ctx,
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

arm_cmsis_nn_status arm_convolve_direct_nhwc_ohwi_f16_acc16(const cmsis_nn_context *ctx,
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
    if (input_dims && filter_dims && input_dims->c != filter_dims->c)
    {
        return arm_convolve_grouped_f16_acc16(ctx,
                                              conv_params,
                                              input_dims,
                                              input_data,
                                              filter_dims,
                                              filter_data,
                                              bias_dims,
                                              bias_data,
                                              output_dims,
                                              output_data,
                                              NULL);
    }
    return arm_nn_convolve_single_group_direct_nhwc_ohwi_f16_acc16(ctx,
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

#endif /* ARM_NN_ENABLE_F16 */
