/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#define ARM_RSQRT_S16_BASE_LUT_SIZE (513)
#define ARM_RSQRT_S16_BASE_STEP_SHIFT (6)
#define ARM_RSQRT_S16_SLOT_SHIFT (7)
#define ARM_RSQRT_S16_SLOT_MASK ((1 << ARM_RSQRT_S16_SLOT_SHIFT) - 1)
#define ARM_RSQRT_S16_INTERP_ROUND_BIAS (1 << (ARM_RSQRT_S16_SLOT_SHIFT - 1))

static inline int32_t arm_rsqrt_s16_round_div2_nearest(const int32_t value)
{
    if (value >= 0)
    {
        return (value + 1) >> 1;
    }

    return -(((-value) + 1) >> 1);
}

static inline int32_t arm_rsqrt_s16_scaled_positive(const int32_t *lut,
                                                    int32_t q_value,
                                                    const int32_t out_mult,
                                                    const int32_t out_shift,
                                                    const bool needs_rescale)
{
    int32_t lut_index =
        (q_value + ((1 << ARM_RSQRT_S16_BASE_STEP_SHIFT) - 1)) >> ARM_RSQRT_S16_BASE_STEP_SHIFT;
    if (lut_index >= ARM_RSQRT_S16_BASE_LUT_SIZE)
    {
        lut_index = ARM_RSQRT_S16_BASE_LUT_SIZE - 1;
    }

    int32_t scaled = lut[lut_index];
    if (needs_rescale)
    {
        scaled = arm_nn_requantize(scaled, out_mult, out_shift);
    }
    scaled = MIN(INT16_MAX, scaled);
    scaled = MAX(INT16_MIN, scaled);
    return scaled;
}

static inline int32_t arm_rsqrt_s16_sample_slot(const int32_t *lut,
                                                const int32_t slot_index,
                                                const int32_t out_mult,
                                                const int32_t out_shift,
                                                const bool needs_rescale)
{
    const int32_t q0 = INT16_MIN + (slot_index << ARM_RSQRT_S16_SLOT_SHIFT);
    const int32_t qmid = q0 + (1 << ARM_RSQRT_S16_BASE_STEP_SHIFT);
    const int32_t q1 = q0 + (1 << ARM_RSQRT_S16_SLOT_SHIFT);

    const int32_t s0 =
        (q0 <= 0) ? INT16_MAX : arm_rsqrt_s16_scaled_positive(lut, q0, out_mult, out_shift, needs_rescale);
    const int32_t sm =
        (qmid <= 0) ? INT16_MAX : arm_rsqrt_s16_scaled_positive(lut, qmid, out_mult, out_shift, needs_rescale);
    const int32_t s1 =
        (q1 <= 0) ? INT16_MAX : arm_rsqrt_s16_scaled_positive(lut, q1, out_mult, out_shift, needs_rescale);

    const int32_t midpoint_interp = (s0 + s1 + 1) >> 1;
    const int32_t midpoint_err = midpoint_interp - sm;
    const int32_t bias = arm_rsqrt_s16_round_div2_nearest(midpoint_err);
    int32_t slot_value = s0 - bias;

    slot_value = MIN(INT16_MAX, slot_value);
    slot_value = MAX(INT16_MIN, slot_value);
    return slot_value;
}

static arm_cmsis_nn_status arm_rsqrt_s16_universal_core(const int16_t *input,
                                                        const int32_t input_offset,
                                                        int16_t *output,
                                                        const int32_t out_offset,
                                                        const int32_t out_mult,
                                                        const int32_t out_shift,
                                                        const bool needs_rescale,
                                                        const int32_t out_activation_min,
                                                        const int32_t out_activation_max,
                                                        const int32_t block_size,
                                                        const int32_t *lut)
{
#if defined(ARM_MATH_MVEI)
    int32_t i = 0;
    for (; i + 3 < block_size; i += 4)
    {
        int32_t y0_buf[4];
        int32_t y1_buf[4];
        int32_t frac_buf[4];

        for (size_t lane = 0; lane < 4; lane++)
        {
            int32_t val = (int32_t) input[i + lane] - input_offset;
            if (val < 0)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (val > 0x7FFF)
            {
                val = 0x7FFF;
            }

            const int32_t shifted_val = val - INT16_MIN;
            const int32_t idx = (shifted_val >> ARM_RSQRT_S16_SLOT_SHIFT) & 0x1FF;
            frac_buf[lane] = shifted_val & ARM_RSQRT_S16_SLOT_MASK;
            y0_buf[lane] = arm_rsqrt_s16_sample_slot(lut, idx, out_mult, out_shift, needs_rescale);
            y1_buf[lane] = arm_rsqrt_s16_sample_slot(lut, idx + 1, out_mult, out_shift, needs_rescale);
        }

        int32x4_t vy0 = vldrwq_s32(y0_buf);
        int32x4_t vy1 = vldrwq_s32(y1_buf);
        int32x4_t vfrac = vldrwq_s32(frac_buf);
        int32x4_t vdelta = vsubq_s32(vy1, vy0);
        int32x4_t vinterp = vmulq_s32(vdelta, vfrac);
        vinterp = vaddq_s32(vinterp, vdupq_n_s32(ARM_RSQRT_S16_INTERP_ROUND_BIAS));
        vinterp = vshrq_n_s32(vinterp, ARM_RSQRT_S16_SLOT_SHIFT);
        int32x4_t vres = vaddq_s32(vy0, vinterp);
        if (out_offset != 0)
        {
            vres = vaddq_s32(vres, vdupq_n_s32(out_offset));
        }
        vres = vminq_s32(vres, vdupq_n_s32(out_activation_max));
        vres = vmaxq_s32(vres, vdupq_n_s32(out_activation_min));
        vstrhq_s32(output + i, vres);
    }

    for (; i < block_size; i++)
    {
#else
    for (int32_t i = 0; i < block_size; i++)
    {
#endif
        int32_t val = (int32_t) input[i] - input_offset;
        if (val < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (val > 0x7FFF)
        {
            val = 0x7FFF;
        }

        const int32_t shifted_val = val - INT16_MIN;
        const int32_t idx = (shifted_val >> ARM_RSQRT_S16_SLOT_SHIFT) & 0x1FF;
        const int32_t frac = shifted_val & ARM_RSQRT_S16_SLOT_MASK;
        const int32_t y0 = arm_rsqrt_s16_sample_slot(lut, idx, out_mult, out_shift, needs_rescale);
        const int32_t y1 = arm_rsqrt_s16_sample_slot(lut, idx + 1, out_mult, out_shift, needs_rescale);
        const int64_t interp_delta =
            (((int64_t) (y1 - y0) * (int64_t) frac) + ARM_RSQRT_S16_INTERP_ROUND_BIAS) >>
            ARM_RSQRT_S16_SLOT_SHIFT;
        int32_t rsqrt_res = (int32_t) ((int64_t) y0 + interp_delta);

        rsqrt_res += out_offset;
        rsqrt_res = MIN(out_activation_max, rsqrt_res);
        rsqrt_res = MAX(out_activation_min, rsqrt_res);
        output[i] = (int16_t) rsqrt_res;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_rsqrt_s16_universal(const int16_t *input,
                                            const int32_t input_offset,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const bool needs_rescale,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size,
                                            const int32_t *lut)
{
    return arm_rsqrt_s16_universal_core(input,
                                        input_offset,
                                        output,
                                        out_offset,
                                        out_mult,
                                        out_shift,
                                        needs_rescale,
                                        out_activation_min,
                                        out_activation_max,
                                        block_size,
                                        lut);
}

arm_cmsis_nn_status arm_rsqrt_s16_per_op(const int16_t *input,
                                         const int32_t input_offset,
                                         int16_t *output,
                                         const int32_t out_offset,
                                         const int32_t out_activation_min,
                                         const int32_t out_activation_max,
                                         const int32_t block_size,
                                         const int16_t *lut)
{
#if defined(ARM_MATH_MVEI)
    int32_t i = 0;
    for (; i + 3 < block_size; i += 4)
    {
        uint16_t idx_buf[8] = {0};
        uint16_t idx_next_buf[8] = {0};
        int32_t frac_buf[4];

        for (size_t lane = 0; lane < 4; lane++)
        {
            int32_t value = (int32_t) input[i + lane] - input_offset;
            if (value < 0)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (value > 0x7FFF)
            {
                value = 0x7FFF;
            }

            const uint16_t index = (uint16_t) (256 + (value >> ARM_RSQRT_S16_SLOT_SHIFT));
            idx_buf[lane] = index;
            idx_next_buf[lane] = (uint16_t) (index + 1);
            frac_buf[lane] = value & ARM_RSQRT_S16_SLOT_MASK;
        }

        uint16x8_t vidx = vld1q(idx_buf);
        uint16x8_t vidx_next = vld1q(idx_next_buf);
        int16x8_t vy0h = vldrhq_gather_shifted_offset_s16(lut, vidx);
        int16x8_t vy1h = vldrhq_gather_shifted_offset_s16(lut, vidx_next);
        int32x4_t vy0 = vdupq_n_s32(0);
        int32x4_t vy1 = vdupq_n_s32(0);
        vy0 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy0h, 0), vy0, 0);
        vy0 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy0h, 1), vy0, 1);
        vy0 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy0h, 2), vy0, 2);
        vy0 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy0h, 3), vy0, 3);
        vy1 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy1h, 0), vy1, 0);
        vy1 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy1h, 1), vy1, 1);
        vy1 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy1h, 2), vy1, 2);
        vy1 = vsetq_lane_s32((int32_t) vgetq_lane_s16(vy1h, 3), vy1, 3);
        int32x4_t vfrac = vldrwq_s32(frac_buf);
        int32x4_t vdelta = vsubq_s32(vy1, vy0);
        int32x4_t vinterp = vmulq_s32(vdelta, vfrac);
        vinterp = vaddq_s32(vinterp, vdupq_n_s32(ARM_RSQRT_S16_INTERP_ROUND_BIAS));
        vinterp = vshrq_n_s32(vinterp, ARM_RSQRT_S16_SLOT_SHIFT);
        int32x4_t vres = vaddq_s32(vy0, vinterp);
        if (out_offset != 0)
        {
            vres = vaddq_s32(vres, vdupq_n_s32(out_offset));
        }
        vres = vminq_s32(vres, vdupq_n_s32(out_activation_max));
        vres = vmaxq_s32(vres, vdupq_n_s32(out_activation_min));
        vstrhq_s32(output + i, vres);
    }

    for (; i < block_size; i++)
    {
#else
    for (int32_t i = 0; i < block_size; i++)
    {
#endif
        int32_t value = (int32_t) input[i] - input_offset;
        if (value < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (value > 0x7FFF)
        {
            value = 0x7FFF;
        }

        const uint16_t index = (uint16_t) (256 + (value >> ARM_RSQRT_S16_SLOT_SHIFT));
        const int32_t frac = value & ARM_RSQRT_S16_SLOT_MASK;
        const int32_t y0 = lut[index];
        const int32_t y1 = lut[index + 1];
        int32_t rsqrt_res =
            y0 + (((y1 - y0) * frac + (1 << (ARM_RSQRT_S16_SLOT_SHIFT - 1))) >> ARM_RSQRT_S16_SLOT_SHIFT);

        rsqrt_res += out_offset;
        rsqrt_res = MIN(out_activation_max, rsqrt_res);
        rsqrt_res = MAX(out_activation_min, rsqrt_res);
        output[i] = (int16_t) rsqrt_res;
    }

    return ARM_CMSIS_NN_SUCCESS;
}
