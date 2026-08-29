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
 * Title:        arm_squared_difference_s8
 * Description:  Elementwise squared difference w/ support for broadcasting and scalar
 *
 * $Date:        12 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; the quantization parameters and the output
 * requantization parameters are locals of the enclosing function. SCALAR_1 broadcasts one element
 * of input 1 against a run of input 2, SCALAR_2 one element of input 2 against a run of input 1. */
#define ARM_SQDIFF_S8_FULL(a, b, o, n)                                                                                 \
    arm_elementwise_squared_difference_s8((a),                                                                         \
                                          (b),                                                                         \
                                          input1_offset,                                                               \
                                          input1_mult,                                                                 \
                                          input1_shift,                                                                \
                                          input2_offset,                                                               \
                                          input2_mult,                                                                 \
                                          input2_shift,                                                                \
                                          left_shift,                                                                  \
                                          (o),                                                                         \
                                          out_offset,                                                                  \
                                          out_mult,                                                                    \
                                          out_shift,                                                                   \
                                          out_activation_min,                                                          \
                                          out_activation_max,                                                          \
                                          (n))
#define ARM_SQDIFF_S8_SCALAR_1(s, v, o, n)                                                                             \
    arm_squared_difference_scalar_s8((s),                                                                              \
                                     (v),                                                                              \
                                     input1_offset,                                                                    \
                                     input1_mult,                                                                      \
                                     input1_shift,                                                                     \
                                     input2_offset,                                                                    \
                                     input2_mult,                                                                      \
                                     input2_shift,                                                                     \
                                     left_shift,                                                                       \
                                     (o),                                                                              \
                                     out_offset,                                                                       \
                                     out_mult,                                                                         \
                                     out_shift,                                                                        \
                                     out_activation_min,                                                               \
                                     out_activation_max,                                                               \
                                     (n))
#define ARM_SQDIFF_S8_SCALAR_2(s, v, o, n)                                                                             \
    arm_squared_difference_scalar_s8((s),                                                                              \
                                     (v),                                                                              \
                                     input2_offset,                                                                    \
                                     input2_mult,                                                                      \
                                     input2_shift,                                                                     \
                                     input1_offset,                                                                    \
                                     input1_mult,                                                                      \
                                     input1_shift,                                                                     \
                                     left_shift,                                                                       \
                                     (o),                                                                              \
                                     out_offset,                                                                       \
                                     out_mult,                                                                         \
                                     out_shift,                                                                        \
                                     out_activation_min,                                                               \
                                     out_activation_max,                                                               \
                                     (n))

/*
 * s8 elementwise squared difference w/ support for broadcasting and scalar
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_squared_difference_s8(const int8_t *input1_data,
                                              const cmsis_nn_dims *input1_dims,
                                              const int8_t *input2_data,
                                              const cmsis_nn_dims *input2_dims,
                                              const int32_t input1_offset,
                                              const int32_t input1_mult,
                                              const int32_t input1_shift,
                                              const int32_t input2_offset,
                                              const int32_t input2_mult,
                                              const int32_t input2_shift,
                                              const int32_t left_shift,
                                              int8_t *output_data,
                                              const cmsis_nn_dims *output_dims,
                                              const int32_t out_offset,
                                              const int32_t out_mult,
                                              const int32_t out_shift,
                                              const int32_t out_activation_min,
                                              const int32_t out_activation_max)
{
    if (!input1_data || !input2_data || !output_data || !input1_dims || !input2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input1_dims, input2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ARM_NN_BROADCAST_WALK_NHWC(int8_t,
                               int8_t,
                               input1_data,
                               input1_dims,
                               input2_data,
                               input2_dims,
                               output_data,
                               output_dims,
                               ARM_SQDIFF_S8_FULL,
                               ARM_SQDIFF_S8_SCALAR_1,
                               ARM_SQDIFF_S8_SCALAR_2);

    return ARM_CMSIS_NN_SUCCESS;
}

#undef ARM_SQDIFF_S8_FULL
#undef ARM_SQDIFF_S8_SCALAR_1
#undef ARM_SQDIFF_S8_SCALAR_2

/**
 * @} end of Doxygen group
 */