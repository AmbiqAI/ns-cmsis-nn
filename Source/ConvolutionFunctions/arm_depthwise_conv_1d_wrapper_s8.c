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
 * Title:        arm_depthwise_conv_1d_wrapper_s8.c
 * Description:  Wrapper API to select appropriate depthwise conv API based
 *               on dimensions.
 *
 * $Date:        8 October 2025
 * $Revision:    V.2.2.1
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
 * @addtogroup NNConv
 * @{
 */

arm_cmsis_nn_status arm_depthwise_conv_1d_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_context *weight_sum_ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output)
{
    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;
    if (input_dims->w == 1) {
        status = arm_depthwise_conv_1d_valid_s1_d1_s8(
            ctx,
            weight_sum_ctx,
            dw_conv_params,
            quant_params,
            &(cmsis_nn_dims){ .n=input_dims->n, .h=input_dims->w, .w=input_dims->h, .c=input_dims->c },
            input,
            &(cmsis_nn_dims){ .n=filter_dims->n, .h=filter_dims->w, .w=filter_dims->h, .c=filter_dims->c },
            filter,
            bias_dims,
            bias,
            &(cmsis_nn_dims){ .n=output_dims->n, .h=output_dims->w, .w=output_dims->h, .c=output_dims->c },
            output
            );
    }
    else {
        status = arm_depthwise_conv_1d_valid_s1_d1_s8(ctx,
            weight_sum_ctx,
            dw_conv_params,
            quant_params,
            input_dims,
            input,
            filter_dims,
            filter,
            bias_dims,
            bias,
            output_dims,
            output);
    }
    return status;
}