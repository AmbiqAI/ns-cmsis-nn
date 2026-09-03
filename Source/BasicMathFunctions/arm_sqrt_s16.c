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

#if defined(ARM_MATH_MVEI)
// One 8-lane block: gather LUT base/next, widen the slope*offset products to
// int32 for the rounded interpolation, and narrow back once. All lanes are
// computed unpredicated; the caller decides which results reach memory.
// Inactive tail lanes load as zero, which indexes lut[256]/lut[257] - always
// in range for the 513-entry table.
static inline int16x8_t arm_sqrt_block_s16(const int16x8_t val, const int16_t *sqrt_lut)
{
    /* index = 256 + (value >> 7) */
    const int16x8_t idx = vaddq_n_s16(vshrq_n_s16(val, 7), 256);
    const int16x8_t idx_next = vaddq_n_s16(idx, 1);

    /* offset = value & 0x7f */
    const int16x8_t offset = vandq_s16(val, vdupq_n_s16(0x7f));

    /* Gather base and next from LUT */
    const int16x8_t base = vldrhq_gather_shifted_offset_s16(sqrt_lut, (uint16x8_t)idx);
    const int16x8_t next = vldrhq_gather_shifted_offset_s16(sqrt_lut, (uint16x8_t)idx_next);

    /* slope = next - base */
    const int16x8_t slope = vsubq_s16(next, base);

    /* result = base + ((slope * offset + 64) >> 7) */
    int32x4_t acc_lo = vmullbq_int_s16(slope, offset);
    int32x4_t acc_hi = vmulltq_int_s16(slope, offset);
    acc_lo = vaddq_n_s32(acc_lo, 64);
    acc_hi = vaddq_n_s32(acc_hi, 64);
    acc_lo = vshrq_n_s32(acc_lo, 7);
    acc_hi = vshrq_n_s32(acc_hi, 7);

    return vaddq_s16(base, vmovntq_s32(vmovnbq_s32(vdupq_n_s16(0), acc_lo), acc_hi));
}
#endif

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
    // Full 8-lane blocks run unpredicated; at most one predicated block
    // handles the tail, OUTSIDE any loop (the arm_memset_f16 shape, as in
    // arm_hard_swish_f16). This is deliberate and load-bearing, not a style
    // choice: a vctp16q loop around this widening body miscompiles under
    // GCC's implicit-tail-predication conversion (observed with Arm GNU
    // Toolchain 14.2.Rel1 and 15.2.Rel1 at both -O2 and -Ofast, which turned
    // it into dlstp.16/letp). In an architecturally tail-predicated loop,
    // predication is byte-granular across EVERY vector instruction in the
    // body regardless of its element size, so on a partial tail the
    // .32-width vmullbq/vmulltq/vaddq_n_s32/vshrq_n_s32 intermediates here
    // get the upper bytes of their 32-bit lanes masked and the last elements
    // are corrupted for any size not a multiple of 8 (caught on FVP
    // Corstone-300). With no vctp in a loop there is nothing for the dlstp
    // conversion to convert.
    int32_t i = 0;
    for (; i <= block_size - 8; i += 8)
    {
        const int16x8_t val = vldrhq_s16(&input[i]);
        vstrhq_s16(&output[i], arm_sqrt_block_s16(val, sqrt_lut));
    }
    if (i < block_size)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(block_size - i));
        const int16x8_t val = vldrhq_z_s16(&input[i], p);
        vstrhq_p_s16(&output[i], arm_sqrt_block_s16(val, sqrt_lut), p);
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
