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
 * Title:        arm_batch_to_space_nd_s8.c
 * Description:  Convert batch to space for s8 tensors
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
 * Convert batch to space for s8 tensors.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status
arm_batch_to_space_nd_s8(
    const int8_t *input_data,
    const cmsis_nn_dims *input_dims,
    const cmsis_nn_tile *block_shape,
    const cmsis_nn_dims *crop, // n->top, h->left, w->bottom, c->right
    int8_t *output_data,
    const cmsis_nn_dims *output_dims
) {
    // Validate input
    if (!input_data || !input_dims || !block_shape || !crop ||
        !output_data || !output_dims) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Check block shape values
    const int32_t bH = block_shape->h;
    const int32_t bW = block_shape->w;
    if (bH <= 0 || bW <= 0) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Check cropping values
    const int32_t crop_top    = crop->n;
    const int32_t crop_left   = crop->h;
    const int32_t crop_bottom = crop->w;
    const int32_t crop_right  = crop->c;
    if (crop_top < 0 || crop_left < 0 || crop_bottom < 0 || crop_right < 0) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t IN = input_dims->n;
    const int32_t IH = input_dims->h;
    const int32_t IW = input_dims->w;
    const int32_t C  = input_dims->c;

    // Check output dimensions
    const int32_t prodB = bH * bW;
    if ((IN % prodB) != 0) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t ON = IN / (bH * bW);
    const int32_t OH = IH * bH - crop_top - crop_bottom;
    const int32_t OW = IW * bW - crop_left - crop_right;
    const int32_t OC = C;

    if (OH < 0 || OW < 0) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (output_dims->n != ON || output_dims->h != OH ||
        output_dims->w != OW || output_dims->c != OC) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t in_n = 0; in_n < IN; ++in_n) {
        const int32_t out_n          = in_n % ON;
        const int32_t spatial_offset = in_n / ON;
        const int32_t kh             = spatial_offset / bW;
        const int32_t kw             = spatial_offset % bW;

        for (int32_t ih = 0; ih < IH; ++ih) {
            const int32_t oh_base = ih * bH + kh - crop_top;
            if ((unsigned)oh_base >= (unsigned)OH) continue;

            for (int32_t iw = 0; iw < IW; ++iw) {
                const int32_t ow_base = iw * bW + kw - crop_left;
                if ((unsigned)ow_base >= (unsigned)OW) continue;

                size_t in_pix  = (((size_t)in_n * IH + (size_t)ih) * IW + (size_t)iw) * (size_t)C;
                size_t out_pix = (((size_t)out_n * OH + (size_t)oh_base) * OW + (size_t)ow_base) * (size_t)C;

                arm_memcpy_s8(output_data + out_pix, input_data + in_pix, (uint32_t)C);
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Reshape group
 */
