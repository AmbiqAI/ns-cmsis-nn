/*
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Copyright (c) 2026 Ambiq Micro, Inc.
 */

/* Keep grouped dispatch in its own translation unit so the compiler preserves the
 * single-group convolution loops used by existing models. */
#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F32
    #include "Internal/arm_conv_opt_f32.h"
    #include "Internal/arm_nn_activation_flt.h"
    #include "arm_nnfunctions.h"
    #include "arm_nnsupportfunctions.h"

__STATIC_INLINE bool arm_convolve_nhwc_f32_grouped_args_ok(const cmsis_nn_conv_params_f32 *conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float32_t *input_data,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const float32_t *filter_data,
                                                           const cmsis_nn_dims *output_dims,
                                                           const float32_t *output_data)
{
    return conv_params && input_dims && filter_dims && output_dims && input_data && filter_data && output_data &&
        arm_nn_conv_flt_grouped_dims_ok(&conv_params->stride,
                                        &conv_params->padding,
                                        &conv_params->dilation,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        sizeof(float32_t),
                                        4,
                                        conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED);
}

/* Clip wide grouped coordinates before taking any tensor offsets. */
__STATIC_FORCEINLINE void arm_conv_f32_tap_span(int64_t base,
                                                int32_t dilation,
                                                int32_t kernel,
                                                int32_t input,
                                                int32_t *first,
                                                int32_t *end,
                                                uint32_t *coordinate)
{
    /* Validated nonnegative int32 padding gives base >= -INT32_MAX. Thus both
     * clipping distances fit uint32, even when coordinate construction needs int64. */
    const uint32_t lo = base < 0 ? ((uint32_t)(-base) - 1u) / (uint32_t)dilation + 1u : 0;
    const uint32_t hi = input > base ? (uint32_t)((int64_t)input - 1 - base) / (uint32_t)dilation + 1u : 0;
    *first = (int32_t)(lo < (uint32_t)kernel ? lo : (uint32_t)kernel);
    *end = (int32_t)(hi < (uint32_t)kernel ? hi : (uint32_t)kernel);
    *coordinate = *first < *end ? (uint32_t)(base + (int64_t)*first * dilation) : 0;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status arm_grouped_ohwi_f32(const cmsis_nn_conv_params_f32 *conv_params,
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
    const int32_t kernel_ch = filter_dims->c;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dil_h = conv_params->dilation.h;
    const int32_t dil_w = conv_params->dilation.w;
    const int32_t groups = input_c / kernel_ch;
    const int32_t output_ch_per_group = output_c / groups;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float32_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float32_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int64_t in_y0 = (int64_t)out_y * stride_h - pad_h;
            int32_t ky_first, ky_end;
            uint32_t iy_first;
            arm_conv_f32_tap_span(in_y0, dil_h, kernel_h, input_h, &ky_first, &ky_end, &iy_first);
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                const int64_t in_x0 = (int64_t)out_x * stride_w - pad_w;
                float32_t *out_pos = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;

                int32_t kx_first, kx_end;
                uint32_t ix_first;
                arm_conv_f32_tap_span(in_x0, dil_w, kernel_w, input_w, &kx_first, &kx_end, &ix_first);

                for (int32_t group = 0; group < groups; ++group)
                {
                    const int32_t in_ch_start = group * kernel_ch;
                    const int32_t oc_end = (group + 1) * output_ch_per_group;
                    for (int32_t oc = group * output_ch_per_group; oc < oc_end; ++oc)
                    {
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                        /* Carry the OHWI accumulator across taps and reduce once per output (#417). */
                        const float32_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * kernel_ch;
                        float32x4_t vacc = vdupq_n_f32(0.0f);
    #else
                        float32_t acc = bias_data ? bias_data[oc] : 0.0f;
                        const float32_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * kernel_ch;
    #endif

                        uint32_t in_y = iy_first;
                        for (int32_t ky = ky_first; ky < ky_end; ++ky, in_y += (uint32_t)dil_h)
                        {
                            uint32_t in_x = ix_first;
                            /* Keep tap addresses live for one column; unroll the channel loop instead. */
    #if defined(__clang__) && !defined(ARM_MATH_MVEF)
        #pragma clang loop unroll(disable)
    #endif
                            for (int32_t kx = kx_first; kx < kx_end; ++kx, in_x += (uint32_t)dil_w)
                            {
                                const size_t k0 = ((size_t)ky * kernel_w + (size_t)kx) * kernel_ch;
                                const float32_t *x =
                                    input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c + in_ch_start;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                                /* Full blocks unpredicated, one predicated tail: no vctp inside a loop. */
                                const float32_t *w_tap = w_oc + k0;
                                int32_t ic = 0;
                                for (; ic + 4 <= kernel_ch; ic += 4)
                                {
                                    vacc = vfmaq(vacc, vld1q(x + ic), vld1q(w_tap + ic));
                                }
                                if (ic < kernel_ch)
                                {
                                    const mve_pred16_t p = vctp32q((uint32_t)(kernel_ch - ic));
                                    vacc = vfmaq_m(vacc, vld1q_z(x + ic, p), vld1q_z(w_tap + ic, p), p);
                                }
    #else
                                for (int32_t ic = 0; ic < kernel_ch; ++ic)
                                {
                                    acc += x[ic] * w_oc[k0 + (size_t)ic];
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
    }
    return ARM_CMSIS_NN_SUCCESS;
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_grouped_f32(const cmsis_nn_context *ctx,
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
    (void)ctx;
    (void)bias_dims;
    if (!arm_convolve_nhwc_f32_grouped_args_ok(
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
    return arm_grouped_ohwi_f32(
        conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
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
    if (input_dims && filter_dims && input_dims->c != filter_dims->c)
    {
        return arm_convolve_grouped_f32(ctx,
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
    return arm_nn_convolve_single_group_nhwc_f32(ctx,
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

arm_cmsis_nn_status arm_convolve_direct_nhwc_ohwi_f32(const cmsis_nn_context *ctx,
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
    if (input_dims && filter_dims && input_dims->c != filter_dims->c)
    {
        return arm_convolve_grouped_f32(ctx,
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
    return arm_nn_convolve_single_group_direct_nhwc_ohwi_f32(ctx,
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

#endif /* ARM_NN_ENABLE_F32 */
