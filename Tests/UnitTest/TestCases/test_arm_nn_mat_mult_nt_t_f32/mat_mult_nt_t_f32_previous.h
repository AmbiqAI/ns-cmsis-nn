/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32
/* arm_nn_mat_mult_nt_t_f32 as it was at 38c6b99e (mat_mult_nt_t_f32_previous.c). */
arm_cmsis_nn_status mp_mat_mult_nt_t_f32(const float32_t *__RESTRICT lhs,
                                         const float32_t *__RESTRICT rhs,
                                         const float32_t *__RESTRICT bias,
                                         float32_t *__RESTRICT dst,
                                         int32_t lhs_rows,
                                         int32_t rhs_rows,
                                         int32_t rhs_cols,
                                         int32_t row_address_offset,
                                         float32_t activation_min,
                                         float32_t activation_max);
#endif
