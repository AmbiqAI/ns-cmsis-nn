/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_sqrt_s16
 * Description:  Elementwise square root (int16, 513-entry LUT with interpolation)
 *
 * $Date:        7 April 2026
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
 * @addtogroup groupElementwise
 * @{
 */

/*
 * s16 elementwise square root using piecewise LUT with linear interpolation.
 *
 * The LUT has 513 entries covering the full int16 range.
 * Index calculation uses the upper 9 bits; the lower 7 bits drive interpolation.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status
arm_sqrt_s16(const int16_t *input, const cmsis_nn_dims *input_dims, int16_t *output, const int16_t *sqrt_lut)
{
    const int32_t block_size = input_dims->n * input_dims->h * input_dims->w * input_dims->c;

#if defined(ARM_MATH_MVEI)
    int32_t loop_cnt = block_size;

    while (loop_cnt > 0)
    {
        const mve_pred16_t p = vctp16q((uint32_t)loop_cnt);
        const int16x8_t val = vldrhq_z_s16(input, p);

        /* index = 256 + (value >> 7) */
        const int16x8_t idx = vaddq_n_s16(vshrq_n_s16(val, 7), 256);
        const int16x8_t idx_next = vaddq_n_s16(idx, 1);

        /* offset = value & 0x7f */
        const int16x8_t offset = vandq_s16(val, vdupq_n_s16(0x7f));

        /* Gather base and next from LUT */
        const int16x8_t base = vldrhq_gather_shifted_offset_z_s16(sqrt_lut, (uint16x8_t)idx, p);
        const int16x8_t next = vldrhq_gather_shifted_offset_z_s16(sqrt_lut, (uint16x8_t)idx_next, p);

        /* slope = next - base */
        const int16x8_t slope = vsubq_s16(next, base);

        /* result = base + ((slope * offset + 64) >> 7) */
        int32x4_t acc_lo = vmullbq_int_s16(slope, offset);
        int32x4_t acc_hi = vmulltq_int_s16(slope, offset);
        acc_lo = vaddq_n_s32(acc_lo, 64);
        acc_hi = vaddq_n_s32(acc_hi, 64);
        acc_lo = vshrq_n_s32(acc_lo, 7);
        acc_hi = vshrq_n_s32(acc_hi, 7);

        const int16x8_t interp = vaddq_s16(base, vmovntq_s32(vmovnbq_s32(vdupq_n_s16(0), acc_lo), acc_hi));

        vstrhq_p_s16(output, interp, p);

        input += 8;
        output += 8;
        loop_cnt -= 8;
    }
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        const int16_t value = input[i];
        const int32_t index = 256 + (value >> 7);
        const int32_t offset = value & 0x7f;
        const int32_t base = (int32_t)sqrt_lut[index];
        const int32_t slope = (int32_t)sqrt_lut[index + 1] - base;
        output[i] = (int16_t)(base + ((slope * offset + 64) >> 7));
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
