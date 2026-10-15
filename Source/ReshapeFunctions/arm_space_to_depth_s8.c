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
 * Title:        arm_space_to_depth_s8.c
 * Description:  Convert space to depth for s8 tensors
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
 * Convert space to depth for s8 tensors.
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_space_to_depth_s8(const int8_t *input_data,
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
    const int32_t C = input_dims->c;

    // Ensure height and width are divisible by block_size
    if ((H % block_size) != 0 || (W % block_size) != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t OH = H / block_size;
    const int32_t OW = W / block_size;
    const int32_t OC = C * block_size * block_size;

    // Validate output dimensions
    if (output_dims->n != N || output_dims->h != OH || output_dims->w != OW || output_dims->c != OC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Copy a horizontal block at once, traversing the input in row order. */
    const uint32_t copy_size = (uint32_t)block_size * (uint32_t)C;
    for (int32_t n = 0; n < N; ++n)
    {
        for (int32_t oh = 0; oh < OH; ++oh)
        {
            const size_t out_row = ((size_t)n * OH + (size_t)oh) * OW * OC;
            for (int32_t kh = 0; kh < block_size; ++kh)
            {
                /* Keep dynamic copy setup compact, especially on Cortex-M0. */
#if defined(__clang__)
    #pragma clang loop unroll(disable)
#endif
                for (int32_t ow = 0; ow < OW; ++ow)
                {
                    const size_t out_offset = out_row + (size_t)ow * OC + (size_t)kh * copy_size;
                    arm_memcpy_s8(output_data + out_offset, input_data, copy_size);
                    input_data += copy_size;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Reshape group
 */
