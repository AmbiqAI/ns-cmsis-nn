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
 * Title:        arm_nn_mat_mult_nt_t_f32.c
 * Description:  Support: matrix multiply (lhs non-transposed, rhs transposed)
 *
 * $Date:        11 November 2026
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_mat_mult_nt_t_flt.h"

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Private
 */

/**
 * @addtogroup groupSupport
 * @{
 */

/* Refer header file for details. The body is Internal/arm_nn_mat_mult_nt_t_flt.h, which a caller may also compile
 * with its own arguments. */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f32(const float32_t *__RESTRICT lhs,
                                             const float32_t *__RESTRICT rhs,
                                             const float32_t *__RESTRICT bias,
                                             float32_t *__RESTRICT dst,
                                             int32_t lhs_rows,
                                             int32_t rhs_rows,
                                             int32_t rhs_cols,
                                             int32_t row_address_offset,
                                             float32_t activation_min,
                                             float32_t activation_max)
{
    return arm_nn_mat_mult_nt_t_f32_inline(
        lhs, rhs, bias, dst, lhs_rows, rhs_rows, rhs_cols, row_address_offset, activation_min, activation_max);
}

/**
 * @} end of groupSupport group
 */

#endif /* ARM_NN_ENABLE_F32 */
