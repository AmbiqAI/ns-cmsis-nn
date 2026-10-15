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
 * Title:        arm_quantize_s8_s8.c
 * Description:  int8 and uint8 requantization
 *
 * $Date:        15 April 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <stdbool.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Quantization
 * @{
 */

/* Positive shifts round identically in both modes. Multiply before dividing so that centered byte * Q31 fits in
 * int64_t, even when the final result exceeds int32_t. Keep this rare path out of the ordinary loop's register
 * allocation. */
static __attribute__((noinline, cold)) void arm_nn_requantize_8bit_wide(const int8_t *input,
                                                                        const bool input_unsigned,
                                                                        int8_t *output,
                                                                        int32_t size,
                                                                        const int32_t effective_scale_multiplier,
                                                                        const int32_t effective_scale_shift,
                                                                        const int32_t input_zeropoint,
                                                                        const int32_t output_zeropoint,
                                                                        const int32_t output_min,
                                                                        const int32_t output_max)
{
    for (int32_t i = 0; i < size; i++)
    {
        const int32_t centered = (input_unsigned ? ((const uint8_t *)input)[i] : input[i]) - input_zeropoint;
        int64_t val = arm_nn_requantize_positive_shift_s64(centered, effective_scale_multiplier, effective_scale_shift);
        val += output_zeropoint;
        val = ARM_NN_CLAMP(val, output_max, output_min);
        if (output_min < 0)
        {
            output[i] = (int8_t)val;
        }
        else
        {
            ((uint8_t *)output)[i] = (uint8_t)val;
        }
    }
}

/* Signedness and output range are compile-time constants in the ordinary loops. */
__STATIC_FORCEINLINE void arm_nn_requantize_8bit(const int8_t *input,
                                                 const bool input_unsigned,
                                                 int8_t *output,
                                                 int32_t size,
                                                 const int32_t effective_scale_multiplier,
                                                 const int32_t effective_scale_shift,
                                                 const int32_t input_zeropoint,
                                                 const int32_t output_zeropoint,
                                                 const int32_t output_min,
                                                 const int32_t output_max)
{
    /* Single-rounding MVE shifts once more than the scale exponent; centered bytes fit through exponent 22. */
    if (effective_scale_shift > 22)
    {
        arm_nn_requantize_8bit_wide(input,
                                    input_unsigned,
                                    output,
                                    size,
                                    effective_scale_multiplier,
                                    effective_scale_shift,
                                    input_zeropoint,
                                    output_zeropoint,
                                    output_min,
                                    output_max);
        return;
    }
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int32_t count = size / 4 + (size % 4 > 0);
    int32x4_t max = vdupq_n_s32(output_max);
    int32x4_t min = vdupq_n_s32(output_min);
    for (int i = 0; i < count; i++)
    {
        const int32_t offset = i * 4;
        mve_pred16_t pred = vctp32q(size);
        size -= 4;
        int32x4_t vals = input_unsigned ? vreinterpretq_s32_u32(vldrbq_z_u32((const uint8_t *)input + offset, pred))
                                        : vldrbq_z_s32(input + offset, pred);
        vals = vaddq_n_s32(vals, -input_zeropoint);
        vals = arm_requantize_mve(vals, effective_scale_multiplier, effective_scale_shift);
        int32x4_t shifted = vaddq_n_s32(vals, output_zeropoint);
        int32x4_t clamped = vminq_s32(vmaxq_s32(shifted, min), max);
        /* The low byte of a value in [output_min, output_max] is the int8_t or uint8_t result */
        vstrbq_p_s32(output + offset, clamped, pred);
    }
#else
    for (int i = 0; i < size; i++)
    {
        int32_t val = (input_unsigned ? ((const uint8_t *)input)[i] : input[i]) - input_zeropoint;
        val = arm_nn_requantize(val, effective_scale_multiplier, effective_scale_shift);
        val += output_zeropoint;
        if (output_min < 0)
        {
            output[i] = ARM_NN_CLAMP(val, output_max, output_min);
        }
        else
        {
            ((uint8_t *)output)[i] = (uint8_t)ARM_NN_CLAMP(val, output_max, output_min);
        }
    }
#endif
}

/*
 * int8_t to int8_t requantization function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_requantize_s8_s8(const int8_t *input,
                                         int8_t *output,
                                         int32_t size,
                                         int32_t effective_scale_multiplier,
                                         int32_t effective_scale_shift,
                                         int32_t input_zeropoint,
                                         int32_t output_zeropoint)
{
    arm_nn_requantize_8bit(input,
                           false,
                           output,
                           size,
                           effective_scale_multiplier,
                           effective_scale_shift,
                           input_zeropoint,
                           output_zeropoint,
                           INT8_MIN,
                           INT8_MAX);
    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * int8_t to uint8_t requantization function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_requantize_s8_u8(const int8_t *input,
                                         uint8_t *output,
                                         int32_t size,
                                         int32_t effective_scale_multiplier,
                                         int32_t effective_scale_shift,
                                         int32_t input_zeropoint,
                                         int32_t output_zeropoint)
{
    arm_nn_requantize_8bit(input,
                           false,
                           (int8_t *)output,
                           size,
                           effective_scale_multiplier,
                           effective_scale_shift,
                           input_zeropoint,
                           output_zeropoint,
                           0,
                           UINT8_MAX);
    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * uint8_t to int8_t requantization function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_requantize_u8_s8(const uint8_t *input,
                                         int8_t *output,
                                         int32_t size,
                                         int32_t effective_scale_multiplier,
                                         int32_t effective_scale_shift,
                                         int32_t input_zeropoint,
                                         int32_t output_zeropoint)
{
    arm_nn_requantize_8bit((const int8_t *)input,
                           true,
                           output,
                           size,
                           effective_scale_multiplier,
                           effective_scale_shift,
                           input_zeropoint,
                           output_zeropoint,
                           INT8_MIN,
                           INT8_MAX);
    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of Dequantization group
 */
