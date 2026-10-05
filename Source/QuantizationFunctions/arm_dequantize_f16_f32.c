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
 * Title:        arm_dequantize_f16_f32.c
 * Description:  Widen float16 to float32, bit-exact
 *
 * $Date:        5 October 2026
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Quantization
 * @{
 */

arm_cmsis_nn_status arm_dequantize_f16_f32(const float16_t *input, float32_t *output, const int32_t block_size)
{
    /* float16_t and uint16_t have the same size and representation; the conversion reads the bits */
    return arm_dequantize_f16_bits_f32((const uint16_t *)(const void *)input, output, block_size);
}

/**
 * @} end of Quantization group
 */

#endif /* ARM_NN_ENABLE_F16 */
