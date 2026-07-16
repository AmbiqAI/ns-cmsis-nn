/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_minmax_f32_common.h
 * Description:  Shared float32 min/max declarations
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_MINMAX_F32_COMMON_H
#define ARM_MINMAX_F32_COMMON_H

#include "arm_nn_compiler.h"
#include "arm_nn_types_flt.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Shared implementation for arm_minimum_f32 / arm_maximum_f32.
 *
 * Defined out-of-line in arm_minmax_f32_common.c (rather than inline in this
 * header) so that source-level test coverage tools (e.g. gcov) attribute
 * hits/lines/branches to the real implementation instead of collapsing them
 * into the call site of the thin wrapper that invokes it.
 *
 * @param[in]  select_max  Non-zero to compute elementwise maximum, zero for minimum.
 *
 * @return  ARM_CMSIS_NN_SUCCESS or ARM_CMSIS_NN_ARG_ERROR.
 */
arm_cmsis_nn_status arm_minmax_f32_impl(const cmsis_nn_context *ctx,
                                        const float32_t *input_1_data,
                                        const cmsis_nn_dims *input_1_dims,
                                        const float32_t *input_2_data,
                                        const cmsis_nn_dims *input_2_dims,
                                        float32_t *output_data,
                                        const cmsis_nn_dims *output_dims,
                                        int32_t select_max);

#ifdef __cplusplus
}
#endif

#endif /* ARM_MINMAX_F32_COMMON_H */
