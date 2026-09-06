/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_depthwise_conv3x3_f16.c
 * Description:  Support: depthwise conv 3x3 kernels for f16
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * Kept for its public signature. The dedicated 3x3 kernel accumulated through memory once per tap and lost
 * to the direct ch_mult == 1 kernel behind arm_depthwise_nhwc_conv_f16 on every shape, so this now forwards
 * there (same tap order; no scratch). See #448.
 */
void arm_nn_depthwise_conv3x3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_h,
                                       int32_t in_w,
                                       const float16_t *__RESTRICT kernel,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t stride_x,
                                       int32_t stride_y,
                                       int32_t pad_x,
                                       int32_t pad_y,
                                       int32_t out_h,
                                       int32_t out_w,
                                       float16_t act_min,
                                       float16_t act_max)
{
    const cmsis_nn_dw_conv_params_f16 params = {.stride = {stride_x, stride_y},
                                                .padding = {pad_x, pad_y},
                                                .dilation = {1, 1},
                                                .ch_mult = 1,
                                                .activation = {act_min, act_max}};
    const cmsis_nn_dims input_dims = {.n = batches, .h = in_h, .w = in_w, .c = in_c};
    const cmsis_nn_dims filter_dims = {.n = 1, .h = 3, .w = 3, .c = in_c};
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = in_c};
    const cmsis_nn_dims output_dims = {.n = batches, .h = out_h, .w = out_w, .c = in_c};

    (void)arm_depthwise_nhwc_conv_f16(
        NULL, &params, &input_dims, x_nhwc, &filter_dims, kernel, &bias_dims, b, &output_dims, out);
}

/**
 * @} end of supportConvolution group
 */

#endif /* ARM_NN_ENABLE_F16 */
