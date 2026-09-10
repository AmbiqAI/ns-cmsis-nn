/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_prelu_s8.c
 * Description:  Parametric ReLU function for int8_t data type
 *
 * $Date:        21 February 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */

/* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; the quantization parameters are locals of the
 * enclosing function. SCALAR_1 broadcasts one input element against a run of alpha, SCALAR_2 one
 * alpha element against a run of the input. */
#define ARM_PRELU_S8_FULL(a, b, o, n)                                                                                  \
    arm_elementwise_prelu_s8((a),                                                                                      \
                             (b),                                                                                      \
                             input_offset,                                                                             \
                             alpha_offset,                                                                             \
                             output_offset,                                                                            \
                             output_multiplier_identity,                                                               \
                             output_shift_identity,                                                                    \
                             output_multiplier_alpha,                                                                  \
                             output_shift_alpha,                                                                       \
                             (o),                                                                                      \
                             (n))
#define ARM_PRELU_S8_SCALAR_1(s, v, o, n)                                                                              \
    arm_prelu_scalar_s8((s),                                                                                           \
                        (v),                                                                                           \
                        true,                                                                                          \
                        input_offset,                                                                                  \
                        alpha_offset,                                                                                  \
                        output_offset,                                                                                 \
                        output_multiplier_identity,                                                                    \
                        output_shift_identity,                                                                         \
                        output_multiplier_alpha,                                                                       \
                        output_shift_alpha,                                                                            \
                        (o),                                                                                           \
                        (n))
#define ARM_PRELU_S8_SCALAR_2(s, v, o, n)                                                                              \
    arm_prelu_scalar_s8((s),                                                                                           \
                        (v),                                                                                           \
                        false,                                                                                         \
                        input_offset,                                                                                  \
                        alpha_offset,                                                                                  \
                        output_offset,                                                                                 \
                        output_multiplier_identity,                                                                    \
                        output_shift_identity,                                                                         \
                        output_multiplier_alpha,                                                                       \
                        output_shift_alpha,                                                                            \
                        (o),                                                                                           \
                        (n))

/*
 * PReLU activation function for int8_t data type.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_prelu_s8(const cmsis_nn_dims *input_dims,
                                 const int8_t *input,
                                 const cmsis_nn_dims *alpha_dims,
                                 const int8_t *alpha,
                                 const int32_t input_offset,
                                 const int32_t alpha_offset,
                                 const int32_t output_offset,
                                 const int32_t output_multiplier_identity,
                                 const int32_t output_shift_identity,
                                 const int32_t output_multiplier_alpha,
                                 const int32_t output_shift_alpha,
                                 const cmsis_nn_dims *output_dims,
                                 int8_t *output)
{
    if (!input || !alpha || !output || !input_dims || !alpha_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_dims, alpha_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* PReLU is elementwise in the input: alpha broadcasts into it, so the output shape is the
     * input shape and alpha may not be wider than the input in any dimension. */
    if (input_dims->n != output_dims->n || input_dims->h != output_dims->h || input_dims->w != output_dims->w ||
        input_dims->c != output_dims->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ARM_NN_BROADCAST_WALK_NHWC(int8_t,
                               int8_t,
                               input,
                               input_dims,
                               alpha,
                               alpha_dims,
                               output,
                               output_dims,
                               ARM_PRELU_S8_FULL,
                               ARM_PRELU_S8_SCALAR_1,
                               ARM_PRELU_S8_SCALAR_2);

    return ARM_CMSIS_NN_SUCCESS;
}

#undef ARM_PRELU_S8_FULL
#undef ARM_PRELU_S8_SCALAR_1
#undef ARM_PRELU_S8_SCALAR_2

/**
 * @} end of Doxygen group
 */
