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
 * Title:        arm_depth_to_space_s8.c
 * Description:  Convert depth to space for s8 tensors
 *
 * $Date:        06 October 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Reshape
 * @{
 */

/*
 * Convert depth to space for s8 tensors.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_depth_to_space_s8(const int8_t *input_data,
                                          const cmsis_nn_dims *input_dims,
                                          const int32_t block_size,
                                          int8_t *output_data,
                                          const cmsis_nn_dims *output_dims)
{
    // Validate input
    if (!input_data || !input_dims || !output_data || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (block_size <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t N = input_dims->n;
    const int32_t H = input_dims->h;
    const int32_t W = input_dims->w;
    const int32_t Ci = input_dims->c;

    // Ensure channels is divisible by (block_size * block_size)
    const int32_t bs2 = block_size * block_size;
    if ((Ci % bs2) != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t Co = Ci / bs2;

    // Validate output dimensions
    const int32_t OH = H * block_size;
    const int32_t OW = W * block_size;
    if (output_dims->n != N || output_dims->h != OH || output_dims->w != OW || output_dims->c != Co)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Each horizontal block contributes one contiguous slice on both sides. */
    const uint32_t copy_size = (uint32_t)block_size * (uint32_t)Co;
    for (int32_t n = 0; n < N; ++n)
    {
        for (int32_t ih = 0; ih < H; ++ih)
        {
            const size_t in_row = ((size_t)n * H + (size_t)ih) * W * Ci;
            for (int32_t kh = 0; kh < block_size; ++kh)
            {
                /* Keep dynamic copy setup compact, especially on Cortex-M0. */
#if defined(__clang__)
    #pragma clang loop unroll(disable)
#endif
                for (int32_t iw = 0; iw < W; ++iw)
                {
                    const size_t in_offset = in_row + (size_t)iw * Ci + (size_t)kh * copy_size;
                    arm_memcpy_s8(output_data, input_data + in_offset, copy_size);
                    output_data += copy_size;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Reshape group
 */
