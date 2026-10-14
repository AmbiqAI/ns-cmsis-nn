/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_resize_bilinear.h"

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    #include <arm_mve.h>

static inline int32x4_t round_shift(int32x4_t value, int32_t half, int32_t shift)
{
    /* Arithmetic shift rounds downward: negative ties need one less than the positive bias. */
    const int32x4_t bias = vpselq_s32(vdupq_n_s32(half - 1), vdupq_n_s32(half), vcmpltq_n_s32(value, 0));
    return vshlq_s32(vaddq_s32(value, bias), vdupq_n_s32(-shift));
}

static void small_channel_row(const int8_t *top,
                              const int8_t *bottom,
                              int8_t *output,
                              int32_t input_width,
                              int32_t output_width,
                              int32_t channels,
                              int32_t sx,
                              int32_t x_shift,
                              int32_t dy,
                              int32_t ry,
                              int32_t shift)
{
    const int32_t dx = 2 * sx;
    const uint32_t count = (uint32_t)output_width * channels;
    const uint32x4_t last = vdupq_n_u32((uint32_t)input_width - 1);
    for (uint32_t i = 0; i < count; i += 4)
    {
        const mve_pred16_t p = vctp32q(count - i);
        const uint32x4_t index = vidupq_n_u32(i, 1);
        const uint32x4_t x = channels == 2 ? vshrq_n_u32(index, 1) : index;
        const uint32x4_t channel = vandq_u32(index, vdupq_n_u32((uint32_t)channels - 1));
        const uint32x4_t pos = vqsubq_u32(vaddq_n_u32(vshlq_n_u32(x, 1), 1), vdupq_n_u32((uint32_t)sx));
        const uint32x4_t raw_x0 = vshlq_u32(pos, vdupq_n_s32(-x_shift));
        const uint32x4_t x0 = vminq_u32(raw_x0, last);
        const uint32x4_t x1 = vminq_u32(vaddq_n_u32(x0, 1), last);
        const uint32x4_t rem =
            vpselq_u32(vandq_u32(pos, vdupq_n_u32((uint32_t)dx - 1)), vdupq_n_u32(0), vcmphiq_u32(last, raw_x0));
        const int32x4_t rx = vreinterpretq_s32_u32(rem);
        const uint32x4_t left = vaddq_u32(vmulq_n_u32(x0, (uint32_t)channels), channel);
        const uint32x4_t right = vaddq_u32(vmulq_n_u32(x1, (uint32_t)channels), channel);
        const int32x4_t tl = vldrbq_gather_offset_z_s32(top, left, p);
        const int32x4_t tr = vldrbq_gather_offset_z_s32(top, right, p);
        const int32x4_t bl = vldrbq_gather_offset_z_s32(bottom, left, p);
        const int32x4_t br = vldrbq_gather_offset_z_s32(bottom, right, p);
        /* Keep both horizontal interpolations unrounded. Bounds include the signed differences. */
        const int32x4_t upper = vaddq_s32(vmulq_n_s32(tl, dx), vmulq_s32(vsubq_s32(tr, tl), rx));
        const int32x4_t lower = vaddq_s32(vmulq_n_s32(bl, dx), vmulq_s32(vsubq_s32(br, bl), rx));
        const int32x4_t value = vaddq_s32(vmulq_n_s32(upper, dy), vmulq_n_s32(vsubq_s32(lower, upper), ry));
        vstrbq_p_s32(output + i, round_shift(value, dx * dy / 2, shift), p);
    }
}
#endif

static uint32_t log2_scale(uint32_t scale)
{
    uint32_t shift = 0;
    while (scale > 1)
    {
        shift++;
        scale >>= 1;
    }
    return shift;
}

static int32_t coordinate(uint32_t position, int32_t input_size, uint32_t scale, uint32_t shift, int32_t *remainder)
{
    const uint32_t twice = 2 * position + 1;
    const uint32_t numerator = twice > scale ? twice - scale : 0;
    const uint32_t lower = numerator >> shift;
    if (lower >= (uint32_t)input_size - 1)
    {
        *remainder = 0;
        return input_size - 1;
    }
    *remainder = (int32_t)(numerator & (2 * scale - 1));
    return (int32_t)lower;
}

arm_cmsis_nn_status arm_resize_bilinear_s8_pow2(const cmsis_nn_resize_params *resize_params,
                                                const cmsis_nn_dims *input_dims,
                                                const int8_t *input_data,
                                                const cmsis_nn_dims *output_dims,
                                                int8_t *output_data)
{
    if (!arm_nn_resize_bilinear_valid(resize_params, input_dims, input_data, output_dims, output_data) ||
        !resize_params->half_pixel_centers || resize_params->align_corners || output_dims->h % input_dims->h ||
        output_dims->w % input_dims->w)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const uint32_t sy = (uint32_t)(output_dims->h / input_dims->h);
    const uint32_t sx = (uint32_t)(output_dims->w / input_dims->w);
    if (!sx || !sy || (sx & (sx - 1)) || (sy & (sy - 1)) || (uint64_t)sx * sy > (1U << 21))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const uint32_t x_shift = log2_scale(sx) + 1;
    const uint32_t y_shift = log2_scale(sy) + 1;
    const uint32_t shift = x_shift + y_shift;
    const int32_t dx = (int32_t)(2 * sx);
    const int32_t dy = (int32_t)(2 * sy);
    const int32_t half = dx * dy / 2;
    const int32_t channels = input_dims->c;
    const int32_t row_stride = input_dims->w * channels;
    const int32_t batch_stride = input_dims->h * row_stride;
    for (int32_t batch = 0; batch < input_dims->n; batch++)
    {
        const int8_t *input = input_data + batch * batch_stride;
        for (int32_t y = 0; y < output_dims->h; y++)
        {
            int32_t ry;
            const int32_t y0 = coordinate((uint32_t)y, input_dims->h, sy, y_shift, &ry);
            const int32_t y1 = y0 + (y0 < input_dims->h - 1);
            const int8_t *top = input + y0 * row_stride;
            const int8_t *bottom = input + y1 * row_stride;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
            if (channels <= 2)
            {
                small_channel_row(top,
                                  bottom,
                                  output_data,
                                  input_dims->w,
                                  output_dims->w,
                                  channels,
                                  (int32_t)sx,
                                  (int32_t)x_shift,
                                  dy,
                                  ry,
                                  (int32_t)shift);
                output_data += output_dims->w * channels;
                continue;
            }
#endif
            for (int32_t x = 0; x < output_dims->w; x++)
            {
                int32_t rx;
                const int32_t x0 = coordinate((uint32_t)x, input_dims->w, sx, x_shift, &rx);
                const int32_t x1 = x0 + (x0 < input_dims->w - 1);
                const int8_t *p00 = top + x0 * channels;
                const int8_t *p01 = top + x1 * channels;
                const int8_t *p10 = bottom + x0 * channels;
                const int8_t *p11 = bottom + x1 * channels;
                const int32_t w00 = (dy - ry) * (dx - rx);
                const int32_t w01 = (dy - ry) * rx;
                const int32_t w10 = ry * (dx - rx);
                const int32_t w11 = ry * rx;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
                for (uint32_t c = 0; c < (uint32_t)channels; c += 4)
                {
                    const mve_pred16_t p = vctp32q((uint32_t)channels - c);
                    int32x4_t value = vmulq_n_s32(vldrbq_z_s32(p00 + c, p), w00);
                    value = vaddq_s32(value, vmulq_n_s32(vldrbq_z_s32(p01 + c, p), w01));
                    value = vaddq_s32(value, vmulq_n_s32(vldrbq_z_s32(p10 + c, p), w10));
                    value = vaddq_s32(value, vmulq_n_s32(vldrbq_z_s32(p11 + c, p), w11));
                    vstrbq_p_s32(output_data + c, round_shift(value, half, (int32_t)shift), p);
                }
#else
                for (int32_t c = 0; c < channels; c++)
                {
                    const int32_t value = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
                    const uint32_t magnitude = (uint32_t)(value < 0 ? -value : value);
                    const int32_t rounded = (int32_t)((magnitude + (uint32_t)half) >> shift);
                    output_data[c] = (int8_t)(value < 0 ? -rounded : rounded);
                }
#endif
                output_data += channels;
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
