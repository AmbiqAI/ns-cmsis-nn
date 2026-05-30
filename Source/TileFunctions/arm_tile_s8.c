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
 * Title:        arm_tile_s8.c
 * Description:  Tile operator for s8 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <string.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Tile
 * @{
 */

arm_cmsis_nn_status arm_tile_s8(const int8_t *input,
                                const cmsis_nn_tile_params *params,
                                int8_t *output)
{
    if (!input || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *input_shape = params->input_shape;
    const int32_t *multiples = params->multiples;

    /* Compute input_size = product of input_shape */
    int32_t input_size = 1;
    for (int32_t d = 0; d < rank; d++)
    {
        input_size *= input_shape[d];
    }

    /* Copy input into the beginning of output */
    memcpy(output, input, (size_t)input_size * sizeof(int8_t));

    /* Expand each dimension in-place from innermost to outermost.
     * Processing back-to-front avoids overwriting source data. */
    int32_t current_size = input_size;

    for (int32_t d = rank - 1; d >= 0; d--)
    {
        const int32_t m = multiples[d];
        if (m > 1)
        {
            /* chunk_size = one slice along dim d with inner dims already tiled */
            int32_t tiled_inner = 1;
            for (int32_t dd = d + 1; dd < rank; dd++)
            {
                tiled_inner *= input_shape[dd] * multiples[dd];
            }
            const int32_t chunk_size = input_shape[d] * tiled_inner;
            const int32_t num_outer = current_size / chunk_size;

            /* Expand each chunk into m copies, processing back-to-front */
            for (int32_t c = num_outer - 1; c >= 0; c--)
            {
                const int32_t src_off = c * chunk_size;
                const int32_t dst_off = c * m * chunk_size;
                for (int32_t t = m - 1; t > 0; t--)
                {
                    memcpy(output + dst_off + t * chunk_size,
                           output + src_off,
                           (size_t)chunk_size * sizeof(int8_t));
                }
                /* Move original chunk to final position */
                if (dst_off != src_off)
                {
                    memmove(output + dst_off, output + src_off,
                            (size_t)chunk_size * sizeof(int8_t));
                }
            }
            current_size *= m;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Tile group
 */
