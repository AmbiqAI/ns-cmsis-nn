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
 * Title:        arm_batch_matmul_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for batch matmul s8 layer function.
 *
 * $Date:        19 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

/**
 *  @ingroup FC
 */

/**
 * @addtogroup GetBufferSizeFC
 * @{
 */

int32_t arm_batch_matmul_s8_get_buffer_size_dsp(const cmsis_nn_dims *input_rhs_dims)
{
    (void)input_rhs_dims;
    return 0;
}

int32_t arm_batch_matmul_s8_get_buffer_size_mve(const cmsis_nn_dims *input_rhs_dims)
{
    // Computed in 64 bits so that a row count large enough to overflow the int32_t byte count cannot wrap past
    // the range check below (mirrors the guard in arm_batch_matmul_s8()).
    const int64_t required_bytes = (int64_t)input_rhs_dims->w * (int64_t)sizeof(int32_t);

    if ((input_rhs_dims->w < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return (int32_t)required_bytes;
}

int32_t arm_batch_matmul_s8_get_buffer_size(const cmsis_nn_dims *input_rhs_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_batch_matmul_s8_get_buffer_size_mve(input_rhs_dims);
#else
    return arm_batch_matmul_s8_get_buffer_size_dsp(input_rhs_dims);
#endif
}

/**
 * @} end of GetBufferSizeFC group
 */
