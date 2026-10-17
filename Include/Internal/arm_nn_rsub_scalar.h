/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_rsub_scalar.h
 * Description:  Vector minus scalar entry points behind the broadcast
 *               arm_sub_s8 and arm_sub_s16 kernels.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_RSUB_SCALAR_H
#define ARM_NN_RSUB_SCALAR_H

#include "arm_nn_types.h"

#include <stdint.h>

/**
 * @brief Vector minus scalar: arm_sub_scalar_s8() with the operands swapped. input_1_vect is the vector and
 *        input_2_vect the scalar; every other parameter is as for arm_sub_scalar_s8().
 */
arm_cmsis_nn_status arm_rsub_scalar_s8(const int8_t *input_1_vect,
                                       const int8_t *input_2_vect,
                                       const int32_t input_1_offset,
                                       const int32_t input_1_mult,
                                       const int32_t input_1_shift,
                                       const int32_t input_2_offset,
                                       const int32_t input_2_mult,
                                       const int32_t input_2_shift,
                                       const int32_t left_shift,
                                       int8_t *output,
                                       const int32_t out_offset,
                                       const int32_t out_mult,
                                       const int32_t out_shift,
                                       const int32_t out_activation_min,
                                       const int32_t out_activation_max,
                                       const int32_t block_size);

/**
 * @brief Vector minus scalar: arm_sub_scalar_s16() with the operands swapped. input_1_vect is the vector and
 *        input_2_vect the scalar; every other parameter is as for arm_sub_scalar_s16().
 */
arm_cmsis_nn_status arm_rsub_scalar_s16(const int16_t *input_1_vect,
                                        const int16_t *input_2_vect,
                                        const int32_t input_1_offset,
                                        const int32_t input_1_mult,
                                        const int32_t input_1_shift,
                                        const int32_t input_2_offset,
                                        const int32_t input_2_mult,
                                        const int32_t input_2_shift,
                                        const int32_t left_shift,
                                        int16_t *output,
                                        const int32_t out_offset,
                                        const int32_t out_mult,
                                        const int32_t out_shift,
                                        const int32_t out_activation_min,
                                        const int32_t out_activation_max,
                                        const int32_t block_size);

#endif /* ARM_NN_RSUB_SCALAR_H */
