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
 * Title:        arm_quantize_f32_s16.c
 * Description:  float32 to int16 Quantization
 *
 * $Date:        15 April 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include <math.h>
#include <string.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Quantization
 * @{
 */

/*
 * float32_t to int16_t quantization function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status
arm_quantize_f32_s16(const float *input, int16_t *output, int32_t size, int32_t zero_point, float scale)
{
    /* The zero point must lie within the output type: the shifted clamp bounds below assume it, and a larger
     * value could not be reached by any output anyway. */
    if (zero_point < INT16_MIN || zero_point > INT16_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t count = (size + 3) / 4;
    float mul_scale = 1.0f / scale;
    /* Clamp before the zero point is added: VCVTA saturates an out-of-range value to INT32_MAX/MIN, and adding
     * the zero point to that would wrap. The bounds are shifted by the zero point instead. */
    int32x4_t max = vdupq_n_s32(INT16_MAX - zero_point);
    int32x4_t min = vdupq_n_s32(INT16_MIN - zero_point);
    for (int i = 0; i < count; i++)
    {
        mve_pred16_t pred = vctp32q(size);
        size -= 4;
        float32x4_t vals = vldrwq_z_f32(input, pred);
        /* VCVTA rounds to nearest with ties away from zero, matching roundf() on the scalar path and the
         * TensorFlow Lite reference. A preceding VRNDN would resolve ties to even first. This leg multiplies by
         * the rounded reciprocal of the scale (Helium has no vector divide), so for a scale whose reciprocal is
         * inexact a value within one ULP of a tie can still land on the other side of it than the dividing
         * scalar leg. */
        int32x4_t rounded = vcvtaq_s32_f32(vmulq_n_f32(vals, mul_scale));
        int32x4_t clamped = vminq_s32(vmaxq_s32(rounded, min), max);
        int32x4_t shifted = vaddq_n_s32(clamped, zero_point);
        vstrhq_p_s32(output, shifted, pred);
        input += 4;
        output += 4;
    }
#else
    for (int i = 0; i < size; i++)
    {
        /* Round half away from zero, then clamp in the float domain so an out-of-range or infinite value never
         * reaches the float-to-int conversion; the bounds are shifted by the zero point so the add cannot wrap.
         * NaN is tested on its bit pattern, since a float compare is not reliable under -ffinite-math-only, and
         * maps to the zero point, which is what VCVTA produces for it on the Helium leg. */
        uint32_t bits;
        memcpy(&bits, &input[i], sizeof(bits));
        float rounded = ((bits & 0x7fffffffU) > 0x7f800000U) ? 0.0f : roundf(input[i] / scale);
        rounded = fminf(fmaxf(rounded, (float)(INT16_MIN - zero_point)), (float)(INT16_MAX - zero_point));
        output[i] = (int16_t)((int32_t)rounded + zero_point);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Quantization group
 */
