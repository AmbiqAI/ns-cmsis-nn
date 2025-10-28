/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_fully_connected_per_channel_s16
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        3 Feb 2025
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
 * @addtogroup FC
 * @{
 */

/*
 * S16 basic fully-connected and matrix multiplication layer function using per-channel quantization for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_fully_connected_per_channel_s16(
    const cmsis_nn_context *ctx,
    const cmsis_nn_fc_params *fc_params,
    const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims,
    const int16_t *input_data,
    const cmsis_nn_dims *filter_dims,
    const int8_t *kernel,
    const cmsis_nn_dims *bias_dims,
    const int64_t *bias_data,
    const cmsis_nn_dims *output_dims,
    int16_t *output_data
)
{
    (void)bias_dims;

    const int32_t output_ch = output_dims->c;
    const int64_t required_bytes = (int64_t)output_ch * (int64_t)sizeof(int32_t);

    if ((ctx == NULL) || (ctx->buf == NULL) || (required_bytes > INT32_MAX))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t required_size = (int32_t)required_bytes;

    if ((ctx->size != 0) && (ctx->size < required_size))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t *reduced_multiplier = (int32_t *)ctx->buf;

    for (int32_t i = 0; i < output_ch; ++i)
    {
        reduced_multiplier[i] = REDUCE_MULTIPLIER(quant_params->multiplier[i]);
    }

    int32_t batch_cnt = input_dims->n;

    while (batch_cnt)
    {

        arm_nn_vec_mat_mult_t_per_ch_s16(
            input_data,
            kernel,
            bias_data,
            output_data,
            reduced_multiplier,
            quant_params->shift,
            filter_dims->n, /* col_dim or accum_depth */
            output_dims->c, /* row_dim or output_depth */
            fc_params->activation.min,
            fc_params->activation.max
        );

        input_data += filter_dims->n;
        output_data += output_dims->c;
        batch_cnt--;
    }
    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of FC group
 */
