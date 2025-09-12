/*
 * SPDX-FileCopyrightText: Copyright 2010-2020, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_depthwise_conv_nt_t_s8.c
 * Description:  Depthwise convolution on matrices with no padding.
 *
 * $Date:        26 October 2022
 * $Revision:    V.2.0.1
 *
 * Target Processor:  Cortex-M processors with MVE extension.
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * ========================================================================
 * MVE VECTORIZED DEPTHWISE CONVOLUTION MICROKERNEL
 * ========================================================================
 * This function performs depthwise convolution on 4 input matrices (lhs) 
 * with a single kernel matrix (rhs) using MVE vectorized operations.
 * 
 * Key Features:
 * - Processes 4 output positions simultaneously for 4x parallelism
 * - Uses MVE 128-bit vectors for 4x channel parallelism  
 * - Optimized for Cortex-M55 with MVE extension
 * - No padding required (handled by caller)
 * 
 * Memory Layout:
 * - lhs: 4 input matrices, each separated by row_x_col * CH_IN_BLOCK_MVE elements
 * - rhs: Kernel weights in [kernel_positions × channels] layout
 * - out: Output buffer for 4 positions × total_ch channels
 * ========================================================================
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_nt_t_s8(const int32_t* weight_sum_buf,  // Pre-computed weight sums for optimization
                                                  const int8_t *lhs,                // Input data: 4 matrices for 4 output positions
                                                  const int8_t *rhs,                // Kernel weights
                                                  const int32_t input_offset,       // Input quantization offset (unused in MVE path)
                                                  const int32_t active_ch,          // Number of active channels to process (≤16)
                                                  const int32_t total_ch,           // Total number of input channels
                                                  const int32_t *out_shift,         // Output quantization shifts (per channel)
                                                  const int32_t *out_mult,          // Output quantization multipliers (per channel)
                                                  const int32_t out_offset,         // Output quantization offset
                                                  const int32_t activation_min,     // Activation function minimum value
                                                  const int32_t activation_max,     // Activation function maximum value
                                                  const uint16_t row_x_col,         // Kernel size (kernel_x × kernel_y)
                                                  const int32_t *const output_bias, // Bias values (unused in MVE path)
                                                  int8_t *out)                      // Output buffer for 4 positions
{
#if defined(ARM_MATH_MVEI)
    /* ========================================================================
     * MVE IMPLEMENTATION: VECTORIZED DEPTHWISE CONVOLUTION
     * ======================================================================== */
    
    /* Unused parameters in MVE implementation (handled by weight sum approach) */
    (void) input_offset;   // Input offset not needed with pre-computed weight sums
    (void) output_bias;    // Bias handled through weight sum optimization
    
    /* ========================================================================
     * CHANNEL BLOCKING FOR MVE VECTORIZATION
     * ========================================================================
     * Process channels in groups of 4 to match MVE vector width (128-bit = 4 × 32-bit)
     * This ensures maximum utilization of MVE vector operations */
    
    /* Calculate number of 4-channel vector iterations needed */
    int32_t loop_count = (active_ch + 3) / 4;
    
    /* Track remaining channels to process in current iteration */
    uint32_t num_ch_to_process = active_ch;
    
    /* Pointer to pre-computed weight sums for optimization */
    const int32_t *weight_sum_base = weight_sum_buf;


    /* ========================================================================
     * MAIN PROCESSING LOOP: CHANNEL VECTORIZATION
     * ========================================================================
     * Process channels in groups of 4 using MVE vectorized operations.
     * Each iteration processes 4 channels across 4 output positions. */
    
    for (int i_loop_cnt = 0, offset = 0; i_loop_cnt < loop_count;
         num_ch_to_process -= 4, offset += 4, out += 4, i_loop_cnt++)
    {
        /* ========================================================================
         * WEIGHT SUM INITIALIZATION
         * ========================================================================
         * Initialize accumulators with pre-computed weight sums for optimization.
         * This eliminates the need to compute bias and input offset during convolution. */
        
        /* Load pre-computed weight sums for current 4-channel group */
        int32x4_t base = vldrwq_s32(weight_sum_base + offset);
        
        /* Initialize 4 output accumulators (one for each output position) */
        int32x4_t out_0 = base;  // Accumulator for output position 0
        int32x4_t out_1 = base;  // Accumulator for output position 1
        int32x4_t out_2 = base;  // Accumulator for output position 2
        int32x4_t out_3 = base;  // Accumulator for output position 3


        /* ========================================================================
         * INPUT DATA POINTER SETUP
         * ========================================================================
         * Set up pointers to access 4 input matrices and kernel weights.
         * Each input matrix represents one output position and is separated
         * by row_x_col * CH_IN_BLOCK_MVE elements in memory. */
        
        /* Kernel weights pointer for current 4-channel group */
        const int8_t *rhs_0 = rhs + offset;
        
        /* Input data pointers for 4 output positions */
        const int8_t *lhs_0 = lhs + offset;                                    // Input matrix 0 (output position 0)
        const int8_t *lhs_1 = lhs + row_x_col * CH_IN_BLOCK_MVE + offset;     // Input matrix 1 (output position 1)
        const int8_t *lhs_2 = lhs + (row_x_col * CH_IN_BLOCK_MVE * 2) + offset; // Input matrix 2 (output position 2)
        const int8_t *lhs_3 = lhs + (row_x_col * CH_IN_BLOCK_MVE * 3) + offset; // Input matrix 3 (output position 3)

        /* ========================================================================
         * CORE CONVOLUTION LOOP: KERNEL POSITION ITERATION
         * ========================================================================
         * For each kernel position, perform vectorized multiply-accumulate operations
         * across all 4 output positions simultaneously. This is the heart of the
         * convolution computation using MVE vectorized operations. */
        
        for (int i_row_x_col = 0; i_row_x_col < row_x_col; i_row_x_col++)
        {
            /* ========================================================================
             * KERNEL WEIGHT LOADING
             * ========================================================================
             * Load 4 kernel weights for current kernel position and 4-channel group.
             * MVE intrinsic: vldrbq_s32() loads 4 consecutive int8 values and
             * converts them to int32x4 vector for arithmetic operations. */
            
            const int32x4_t ker_0 = vldrbq_s32(rhs_0);

            /* ========================================================================
             * VECTORIZED MULTIPLY-ACCUMULATE FOR 4 OUTPUT POSITIONS
             * ========================================================================
             * Process all 4 output positions in parallel using MVE vectorized operations.
             * Each position performs: accumulator += input_vector * kernel_vector */

            /* Output Position 0: Load input data and multiply-accumulate */
            int32x4_t ip_0 = vldrbq_s32(lhs_0);        // Load 4 input values (int8 -> int32x4)
            out_0 += vmulq_s32(ip_0, ker_0);           // Vectorized multiply-accumulate

            /* Output Position 1: Load input data and multiply-accumulate */
            int32x4_t ip_1 = vldrbq_s32(lhs_1);        // Load 4 input values (int8 -> int32x4)
            out_1 += vmulq_s32(ip_1, ker_0);           // Vectorized multiply-accumulate

            /* Output Position 2: Load input data and multiply-accumulate */
            int32x4_t ip_2 = vldrbq_s32(lhs_2);        // Load 4 input values (int8 -> int32x4)
            out_2 += vmulq_s32(ip_2, ker_0);           // Vectorized multiply-accumulate

            /* Output Position 3: Load input data and multiply-accumulate */
            int32x4_t ip_3 = vldrbq_s32(lhs_3);        // Load 4 input values (int8 -> int32x4)
            out_3 += vmulq_s32(ip_3, ker_0);           // Vectorized multiply-accumulate

            /* ========================================================================
             * POINTER ADVANCEMENT FOR NEXT KERNEL POSITION
             * ========================================================================
             * Advance input pointers to next kernel position for all 4 output positions.
             * Advance kernel pointer to next kernel position. */

            /* Advance input data pointers to next kernel position */
            lhs_0 += CH_IN_BLOCK_MVE;  // Next kernel position in input matrix 0
            lhs_1 += CH_IN_BLOCK_MVE;  // Next kernel position in input matrix 1
            lhs_2 += CH_IN_BLOCK_MVE;  // Next kernel position in input matrix 2
            lhs_3 += CH_IN_BLOCK_MVE;  // Next kernel position in input matrix 3

            /* Advance kernel pointer to next kernel position */
            rhs_0 += total_ch;         // Next kernel position in weight matrix
        }

        /* ========================================================================
         * QUANTIZATION PARAMETER LOADING
         * ========================================================================
         * Load quantization parameters for current 4-channel group.
         * These parameters are used to convert accumulated results back to int8. */
        
        const int32x4_t mult = vldrwq_s32(out_mult);    // Load quantization multipliers
        const int32x4_t shift = vldrwq_s32(out_shift);  // Load quantization shifts
        out_mult += 4;                                   // Advance for next iteration
        out_shift += 4;                                  // Advance for next iteration
        
        /* ========================================================================
         * MVE PREDICATED EXECUTION SETUP
         * ========================================================================
         * Create predicate mask for remaining channels in last iteration.
         * This ensures we only process valid channels when active_ch is not
         * a multiple of 4. */
        
        mve_pred16_t p = vctp32q(num_ch_to_process);

        /* ========================================================================
         * QUANTIZATION AND ACTIVATION FOR OUTPUT POSITION 0
         * ========================================================================
         * Apply quantization and activation functions using MVE vectorized operations. */
        
        out_0 = arm_requantize_mve_32x4(out_0, mult, shift);  // Apply requantization: out = (sum * mult) >> shift
        out_0 = vaddq_n_s32(out_0, out_offset);               // Add output quantization offset
        out_0 = vmaxq_s32(out_0, vdupq_n_s32(activation_min)); // Apply activation minimum (clamp)
        out_0 = vminq_s32(out_0, vdupq_n_s32(activation_max)); // Apply activation maximum (clamp)
        vstrbq_p_s32(out, out_0, p);                          // Store results (predicated store)

        /* ========================================================================
         * QUANTIZATION AND ACTIVATION FOR OUTPUT POSITION 1
         * ======================================================================== */
        
        out_1 = arm_requantize_mve_32x4(out_1, mult, shift);  // Apply requantization
        out_1 = vaddq_n_s32(out_1, out_offset);               // Add output offset
        out_1 = vmaxq_s32(out_1, vdupq_n_s32(activation_min)); // Apply activation minimum
        out_1 = vminq_s32(out_1, vdupq_n_s32(activation_max)); // Apply activation maximum
        vstrbq_p_s32(out + total_ch, out_1, p);               // Store at position 1 offset

        /* ========================================================================
         * QUANTIZATION AND ACTIVATION FOR OUTPUT POSITION 2
         * ======================================================================== */
        
        out_2 = arm_requantize_mve_32x4(out_2, mult, shift);  // Apply requantization
        out_2 = vaddq_n_s32(out_2, out_offset);               // Add output offset
        out_2 = vmaxq_s32(out_2, vdupq_n_s32(activation_min)); // Apply activation minimum
        out_2 = vminq_s32(out_2, vdupq_n_s32(activation_max)); // Apply activation maximum
        vstrbq_p_s32(out + 2 * total_ch, out_2, p);           // Store at position 2 offset

        /* ========================================================================
         * QUANTIZATION AND ACTIVATION FOR OUTPUT POSITION 3
         * ======================================================================== */
        
        out_3 = arm_requantize_mve_32x4(out_3, mult, shift);  // Apply requantization
        out_3 = vaddq_n_s32(out_3, out_offset);               // Add output offset
        out_3 = vmaxq_s32(out_3, vdupq_n_s32(activation_min)); // Apply activation minimum
        out_3 = vminq_s32(out_3, vdupq_n_s32(activation_max)); // Apply activation maximum
        vstrbq_p_s32(out + 3 * total_ch, out_3, p);           // Store at position 3 offset
    }

    /* ========================================================================
     * SUCCESSFUL COMPLETION
     * ========================================================================
     * All 4 output positions have been processed successfully using MVE
     * vectorized operations. Return success status. */
    
    return ARM_CMSIS_NN_SUCCESS;
#else
    /* ========================================================================
     * NON-MVE IMPLEMENTATION
     * ========================================================================
     * This function requires MVE extension. For non-MVE processors, use
     * the general depthwise convolution implementation instead. */
    
    (void)weight_sum_buf;
    (void)lhs;
    (void)rhs;
    (void)input_offset;
    (void)active_ch;
    (void)total_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)row_x_col;
    (void)output_bias;
    (void)out;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of Doxygen group
 */
