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
 * Title:        arm_dequantize_half_bits.c
 * Description:  Widen float16, given as raw IEEE bits, to float32 as the hardware conversion does
 *
 * $Date:        5 October 2026
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

arm_cmsis_nn_status arm_dequantize_f16_bits_f32(const uint16_t *input, float *output, const int32_t block_size)
{
    if (block_size < 0 || ((input == NULL || output == NULL) && block_size != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_nn_dequantize_f16_bits_f32(input, output, block_size);
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Quantization group
 */
