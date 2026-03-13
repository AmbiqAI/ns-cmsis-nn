/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @addtogroup Reduction
 * @{
 */

/**
 * @brief Sum reduction wrapper calling the optimized mean kernels.
 *
 * @details This function reuses arm_mean_s16 logic. In quantized networks, 
 * ReduceSum and Mean are identical operators where the only 
 * difference is the output scaling (multiplier/shift).
 */
arm_cmsis_nn_status arm_reduce_sum_s16(const int16_t *input_data,
                                       const cmsis_nn_dims *input_dims,
                                       const int32_t input_offset,
                                       const cmsis_nn_dims *axis_dims,
                                       int16_t *output_data,
                                       const cmsis_nn_dims *output_dims,
                                       const int32_t out_offset,
                                       const int32_t out_mult,
                                       const int32_t out_shift)
{
    /* * Since you explicitly want to use the existing optimized code:
     * We simply redirect to arm_mean_s16.
     * The Ambiq arm_mean_s16 handles:
     * 1. Dispatching to MVE (Helium) specialized kernels.
     * 2. Dispatching to the Flattened fast-path (suffix reduction).
     * 3. Fallback to the generic scalar reduction.
     */
    return arm_mean_s16(input_data,
                        input_dims,
                        input_offset,
                        axis_dims,
                        output_data,
                        output_dims,
                        out_offset,
                        out_mult,
                        out_shift);
}

/**
 * @} end of Reduction group
 */
