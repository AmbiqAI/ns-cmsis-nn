/*
 * SPDX-FileCopyrightText: Copyright 2010-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
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
 * Title:        arm_convolve_wrapper_s8.c
 * Description:  s8 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 * cmsis-nn to perform the convolution.
 *
 * $Date:        28 September 2026
 * $Revision:    V.2.6.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_dispatch.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if !ARM_NN_ROUTES_MATCH_BUILD
    #error "ARM_NN_ROUTE_* must describe the build that compiles this wrapper"
#endif

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Convolution layer
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_context *weight_sum_ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data)
{
    switch (arm_nn_convolve_wrapper_s8_select(conv_params, input_dims, filter_dims, output_dims))
    {
    case ARM_NN_CONV_S8_ROUTE_1X1_FAST:
        return arm_convolve_1x1_s8_fast(ctx,
                                        weight_sum_ctx,
                                        conv_params,
                                        quant_params,
                                        input_dims,
                                        input_data,
                                        filter_dims,
                                        filter_data,
                                        bias_dims,
                                        bias_data,
                                        output_dims,
                                        output_data);
    case ARM_NN_CONV_S8_ROUTE_1X1:
        return arm_convolve_1x1_s8(ctx,
                                   weight_sum_ctx,
                                   conv_params,
                                   quant_params,
                                   input_dims,
                                   input_data,
                                   filter_dims,
                                   filter_data,
                                   bias_dims,
                                   bias_data,
                                   output_dims,
                                   output_data);
    case ARM_NN_CONV_S8_ROUTE_1_X_N:
        return arm_convolve_1_x_n_s8(ctx,
                                     weight_sum_ctx,
                                     conv_params,
                                     quant_params,
                                     input_dims,
                                     input_data,
                                     filter_dims,
                                     filter_data,
                                     bias_dims,
                                     bias_data,
                                     output_dims,
                                     output_data);
#if defined(ARM_MATH_MVEI)
    case ARM_NN_CONV_S8_ROUTE_1X1_OUT:
        return arm_convolve_1x1_out_s8(ctx,
                                       weight_sum_ctx,
                                       conv_params,
                                       quant_params,
                                       input_dims,
                                       input_data,
                                       filter_dims,
                                       filter_data,
                                       bias_dims,
                                       bias_data,
                                       output_dims,
                                       output_data);
#endif
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    case ARM_NN_CONV_S8_ROUTE_SMALL_CIN:
        return arm_convolve_s8_small_cin(ctx,
                                         weight_sum_ctx,
                                         conv_params,
                                         quant_params,
                                         input_dims,
                                         input_data,
                                         filter_dims,
                                         filter_data,
                                         bias_dims,
                                         bias_data,
                                         NULL,
                                         output_dims,
                                         output_data);
    case ARM_NN_CONV_S8_ROUTE_3X3_C16_S1:
        return arm_convolve_s8_3x3_c16_s1(ctx,
                                          weight_sum_ctx,
                                          conv_params,
                                          quant_params,
                                          input_dims,
                                          input_data,
                                          filter_dims,
                                          filter_data,
                                          bias_dims,
                                          bias_data,
                                          NULL,
                                          output_dims,
                                          output_data);
#endif
    default:
        return arm_convolve_s8(ctx,
                               weight_sum_ctx,
                               conv_params,
                               quant_params,
                               input_dims,
                               input_data,
                               filter_dims,
                               filter_data,
                               bias_dims,
                               bias_data,
                               NULL,
                               output_dims,
                               output_data);
    }
}

/**
 * @} end of NNConv group
 */
