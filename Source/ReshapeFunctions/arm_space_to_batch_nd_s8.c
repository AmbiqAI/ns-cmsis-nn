/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_space_to_batch_nd_s8.c
 * Description:  Convert space to batch for s8 tensors
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
 * Generic space to batch for s8 tensors.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status
arm_space_to_batch_nd_s8(
    const int8_t *input_data,
    const cmsis_nn_dims *input_dims,
    const cmsis_nn_tile *block_shape,
    const cmsis_nn_dims *pad,      // n->top, h->left, w->bottom, c->right
    int8_t *output_data,
    const cmsis_nn_dims *output_dims,
    const int32_t output_offset
) {
    const int B = input_dims->n;
    const int H = input_dims->h;
    const int W = input_dims->w;
    const int C = input_dims->c;

    const int bh = block_shape->h;
    const int bw = block_shape->w;

    const int pad_top    = pad->n;
    const int pad_left   = pad->h;
    const int pad_bottom = pad->w;
    const int pad_right  = pad->c;

    if (bh <= 0 || bw <= 0) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int Hpad = pad_top + H + pad_bottom;
    const int Wpad = pad_left + W + pad_right;

    /* TF SpaceToBatchND contract */
    if ((Hpad % bh) != 0 || (Wpad % bw) != 0) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int Bout = B * bh * bw;
    const int Hout = Hpad / bh;
    const int Wout = Wpad / bw;

    if (output_dims) {
        if (output_dims->n != Bout || output_dims->h != Hout ||
            output_dims->w != Wout || output_dims->c != C) {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    /* 1) Prefill with output zero-point (pad value), like TFLM */
    const uint32_t out_bytes = (uint32_t)Bout * (uint32_t)Hout *
                               (uint32_t)Wout * (uint32_t)C;
    arm_memset_s8(output_data, (int8_t)output_offset, out_bytes);

    /* 2) Scatter-copy valid input pixels to their output locations.
          NHWC layout; ib_h/ib_w are the intra-block offsets. */
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const int ph   = h + pad_top;
            const int oh   = ph / bh;
            const int ib_h = ph % bh;

            for (int w = 0; w < W; ++w) {
                const int pw   = w + pad_left;
                const int ow   = pw / bw;
                const int ib_w = pw % bw;

                /* TFLM batch ordering:
                   out_b = b + B * (ib_w + bw * ib_h) */
                const int out_b = b + B * (ib_w + bw * ib_h);

                const int32_t in_index  = (((b * H + h) * W + w) * C);
                const int32_t out_index = (((out_b * Hout + oh) * Wout + ow) * C);

                arm_memcpy_s8(&output_data[out_index], &input_data[in_index], (uint32_t)C);
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of Reshape group
 */
