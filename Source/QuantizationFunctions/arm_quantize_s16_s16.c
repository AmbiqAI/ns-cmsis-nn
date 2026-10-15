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
 * Title:        arm_quantize_s16_s16.c
 * Description:  int16 to int16 requantization
 *
 * $Date:        15 April 2025
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
 * @addtogroup Quantization
 * @{
 */

/* Keep the uncommon wide arithmetic out of the ordinary loop's register allocation. */
static __attribute__((noinline)) void arm_nn_requantize_s16_wide(const int16_t *input,
                                                                 int16_t *output,
                                                                 const int32_t size,
                                                                 const int32_t multiplier,
                                                                 const int32_t shift,
                                                                 const int32_t input_zeropoint,
                                                                 const int32_t output_zeropoint)
{
    for (int32_t i = 0; i < size; i++)
    {
        const int32_t centered = (int32_t)input[i] - input_zeropoint;
        const int64_t val = arm_nn_requantize_positive_shift_s64(centered, multiplier, shift) + output_zeropoint;
        output[i] = (int16_t)ARM_NN_CLAMP(val, INT16_MAX, INT16_MIN);
    }
}

/*
 * int16_t to int16_t requantization function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_requantize_s16_s16(const int16_t *input,
                                           int16_t *output,
                                           int32_t size,
                                           int32_t effective_scale_multiplier,
                                           int32_t effective_scale_shift,
                                           int32_t input_zeropoint,
                                           int32_t output_zeropoint)
{
    /* Centered int16 values fit the extra single-rounding MVE left shift through exponent 14. */
    if (effective_scale_shift > 14)
    {
        arm_nn_requantize_s16_wide(
            input, output, size, effective_scale_multiplier, effective_scale_shift, input_zeropoint, output_zeropoint);
        return ARM_CMSIS_NN_SUCCESS;
    }
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int32_t count = size / 4 + (size % 4 > 0);
    int32x4_t max = vdupq_n_s32(INT16_MAX);
    int32x4_t min = vdupq_n_s32(INT16_MIN);
    for (int i = 0; i < count; i++)
    {
        const int32_t offset = i * 4;
        mve_pred16_t pred = vctp32q(size);
        size -= 4;
        int32x4_t vals = vldrhq_z_s32(input + offset, pred);
        vals = vaddq_n_s32(vals, -input_zeropoint);
        vals = arm_requantize_mve(vals, effective_scale_multiplier, effective_scale_shift);
        int32x4_t shifted = vaddq_n_s32(vals, output_zeropoint);
        int32x4_t clamped = vminq_s32(vmaxq_s32(shifted, min), max);
        vstrhq_p_s32(output + offset, clamped, pred);
    }
#else
    for (int i = 0; i < size; i++)
    {
        int32_t val = input[i] - input_zeropoint;
        val = arm_nn_requantize(val, effective_scale_multiplier, effective_scale_shift);
        val += output_zeropoint;
        output[i] = ARM_NN_CLAMP(val, INT16_MAX, INT16_MIN);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of Dequantization group
 */
