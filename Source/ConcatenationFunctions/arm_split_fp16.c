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
 * Title:        arm_split_fp16.c
 * Description:  Split fp16 vectors
 *
 * $Date:        22 July 2026
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
 * @addtogroup Concatenation
 * @{
 */

#if defined(ARM_FLOAT16_SUPPORTED)

arm_cmsis_nn_status arm_split_fp16(const float16_t *input_data,
                                   const int32_t input_dims,
                                   const int32_t *input_shape,
                                   const int32_t axis,
                                   const int32_t num_splits,
                                   const int32_t *split_dims,
                                   float16_t *const *output_data)
{
    // Compute "outer" size: product of dims before 'axis'.
    int64_t outer_size = 1;
    for (int32_t d = 0; d < axis; d++)
    {
        outer_size *= input_shape[d];
    }
    // Compute base "inner" size: product of dims after 'axis'.
    int64_t base_inner_size = 1;
    for (int32_t d = axis + 1; d < input_dims; d++) {
        base_inner_size *= input_shape[d];
    }

    // For each outer index...
    for (int k = 0; k < outer_size; k++) {

        // For each output tensor (split)...
        for (int s = 0; s < num_splits; s++) {

            // The number of elements to copy for this output.
            const int copy_size = split_dims[s] * base_inner_size;
            float16_t *out_ptr = output_data[s] + k * copy_size;
            // fp16 is a 2-byte element; split is a pure data-movement op, so the
            // s16 copy is bit-exact for float16 payloads.
            arm_memcpy_s16((int16_t *)out_ptr, (const int16_t *)input_data, copy_size);
            input_data += copy_size;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* defined(ARM_FLOAT16_SUPPORTED) */

/**
 * @} end of Concatenation group
 */
