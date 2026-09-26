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

#if defined(ARM_MATH_MVEI)
    const int32x4_t min = vdupq_n_s32(INT16_MIN);
    const int32x4_t max = vdupq_n_s32(INT16_MAX);

    /* Whole blocks run unpredicated and the remainder is peeled out. Predicating the main
       loop instead costs a VCTP and two VPST per iteration: the compiler folds the INT16
       clamp into VQMOVNB, which then makes LLVM reject the loop for hardware tail
       predication, so neither form of predication comes for free here. */
    int32_t blocks = size >> 2;
    while (blocks-- > 0)
    {
        int32x4_t vals = vldrhq_s32(input);
        vals = vaddq_n_s32(vals, -input_zeropoint);
        vals = arm_requantize_mve(vals, effective_scale_multiplier, effective_scale_shift);
        vals = vaddq_n_s32(vals, output_zeropoint);
        vstrhq_s32(output, vminq_s32(vmaxq_s32(vals, min), max));
        input += 4;
        output += 4;
    }

    const int32_t tail = size & 3;
    if (tail)
    {
        mve_pred16_t pred = vctp32q(tail);
        int32x4_t vals = vldrhq_z_s32(input, pred);
        vals = vaddq_n_s32(vals, -input_zeropoint);
        vals = arm_requantize_mve(vals, effective_scale_multiplier, effective_scale_shift);
        vals = vaddq_n_s32(vals, output_zeropoint);
        vstrhq_p_s32(output, vminq_s32(vmaxq_s32(vals, min), max), pred);
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
