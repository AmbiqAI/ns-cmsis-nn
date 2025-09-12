/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_depthwise_conv_s8_opt.c
 * Description:  Optimized s8 depthwise separable convolution function for
 *               channel multiplier of 1.
 *
 * $Date:        22 March 2023
 * $Revision:    V.3.5.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Optimized s8 depthwise convolution function with constraint that in_channel equals out_channel
 *
 *  Refer prototype header file for details.
 *
 */

arm_cmsis_nn_status arm_depthwise_conv_s8_opt(const cmsis_nn_context *ctx,
                                              const cmsis_nn_context *weight_sum_ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *kernel,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output)
{
    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;

    /* Check depth multiplier is 1 */
    if (input_ch != output_ch)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (ctx->buf == NULL && arm_depthwise_conv_s8_opt_get_buffer_size(input_dims, filter_dims) > 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#ifdef ARM_MATH_DSP
    (void)bias_dims;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t *output_shift = quant_params->shift;
    const int32_t *output_mult = quant_params->multiplier;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_offset = dw_conv_params->output_offset;
    const int32_t input_offset = dw_conv_params->input_offset;
    const int32_t output_activation_min = dw_conv_params->activation.min;
    const int32_t output_activation_max = dw_conv_params->activation.max;
    int16_t *buffer_a = (int16_t *)ctx->buf;

    #ifdef ARM_MATH_MVEI
    /* ========================================================================
     * MVE (M-Profile Vector Extension) OPTIMIZED DEPTHWISE CONVOLUTION
     * ========================================================================
     * This implementation uses vectorized im2col + batch processing for Cortex-M55
     * Key optimizations:
     * 1. Channel blocking: Process channels in blocks of CH_IN_BLOCK_MVE (typically 16)
     * 2. Im2col transformation: Rearrange input data for efficient vectorized operations
     * 3. Batch processing: Process 4 output positions simultaneously
     * 4. MVE vectorization: Use 128-bit vector operations for 4x parallelism
     * ======================================================================== */
    
    /* Pre-computed weight sums buffer for optimization */
    int32_t* weight_sum_buf = (int32_t *)weight_sum_ctx->buf;
    /* Im2col buffer: stores rearranged input data for vectorized processing
     * Layout: [kernel_size × CH_IN_BLOCK_MVE × 4] for batch processing */
    int8_t *lhs_buffer = (int8_t *)buffer_a;
    
    /* Output pointer for current channel block */
    int8_t *out = output;
    
    /* Counter for batch processing - process 4 output positions at once */
    int buffer_count = 0;
    
    /* Total number of kernel positions (kernel_x × kernel_y) */
    const int32_t kernel_size = kernel_x * kernel_y;

    /* ========================================================================
     * CHANNEL BLOCKING FOR MVE VECTORIZATION
     * ========================================================================
     * MVE works best with 128-bit vectors, typically processing 16 int8 values
     * We divide channels into blocks to maximize vectorization efficiency */
    
    /* Calculate number of channel blocks needed */
    const int32_t ch_loop = (input_ch + (CH_IN_BLOCK_MVE - 1)) / CH_IN_BLOCK_MVE;
    
    /* Track remaining channels to process */
    int32_t remaining_ch = output_ch;
    
    /* Number of channels to process in current block (≤ CH_IN_BLOCK_MVE) */
    int32_t active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch);
    remaining_ch -= CH_IN_BLOCK_MVE;

    /* ========================================================================
     * MAIN PROCESSING LOOP: CHANNEL BLOCKS
     * ======================================================================== */
    for (int i_ch = 0; i_ch < ch_loop; i_ch++)
    {
        /* Set output pointer for current channel block */
        out = output + i_ch * CH_IN_BLOCK_MVE;
        
        /* Get input slice for current channel block */
        const int8_t *input_slice = input + (i_ch * CH_IN_BLOCK_MVE);

        /* ========================================================================
         * IM2COL BUFFER CONSTRUCTION
         * ========================================================================
         * For each output position, build an im2col buffer containing all kernel
         * positions. This enables efficient vectorized operations by reorganizing
         * data layout for MVE processing. */

        /* Iterate through all output positions (y, x) */
        for (int i_out_y = 0, base_idx_y = -pad_y; i_out_y < output_y; base_idx_y += stride_y, i_out_y++)
        {
            for (int i_out_x = 0, base_idx_x = -pad_x; i_out_x < output_x; base_idx_x += stride_x, i_out_x++)
            {
                /* ========================================================================
                 * KERNEL POSITION ITERATION
                 * ========================================================================
                 * For current output position, iterate through all kernel positions
                 * and copy corresponding input data into im2col buffer */
                
                for (int i_ker_y = base_idx_y; i_ker_y < base_idx_y + kernel_y; i_ker_y++)
                {
                    for (int i_ker_x = base_idx_x; i_ker_x < base_idx_x + kernel_x; i_ker_x++)
                    {
                        /* ========================================================================
                         * PADDING HANDLING
                         * ========================================================================
                         * Check if current kernel position is within input bounds */
                        
                        if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x)
                        {
                            /* OUT-OF-BOUNDS: Apply zero-padding
                             * Fill with -input_offset to achieve zero-padding effect
                             * This is equivalent to (input_value - input_offset) = 0 */
                            arm_memset_s8(lhs_buffer, (int8_t)-input_offset, (uint32_t)active_ch);
                        }
                        else
                        {
                            /* IN-BOUNDS: Copy actual input data
                             * Copy active_ch channels from input tensor
                             * Address calculation: (y * input_x + x) * input_ch + channel_offset */
                            arm_memcpy_s8(lhs_buffer,
                                          input_slice + (i_ker_y * input_x + i_ker_x) * input_ch,
                                          (uint32_t)active_ch);
                        }
                        
                        /* Advance im2col buffer pointer for next kernel position
                         * Each kernel position contributes CH_IN_BLOCK_MVE channels */
                        lhs_buffer += CH_IN_BLOCK_MVE;
                    }
                }
                
                /* Increment batch counter - we process 4 output positions at once */
                buffer_count++;

                /* ========================================================================
                 * BATCH PROCESSING: 4 OUTPUT POSITIONS AT ONCE
                 * ========================================================================
                 * When we have accumulated 4 output positions in the im2col buffer,
                 * process them all together using optimized MVE vectorized operations.
                 * This provides 4x parallelism and reduces function call overhead. */
                
                if (buffer_count == 4)
                {
                    /* Calculate offset for current channel block */
                    const int32_t block_offset = i_ch * CH_IN_BLOCK_MVE;
                    
                    /* Reset im2col buffer pointer to beginning */
                    lhs_buffer = (int8_t *)buffer_a;

                    /* ========================================================================
                     * MVE VECTORIZED CONVOLUTION KERNEL
                     * ========================================================================
                     * Call optimized MVE kernel that processes 4 output positions
                     * simultaneously using vectorized operations. This is the core
                     * performance-critical section of the algorithm. */
                    
                    arm_nn_depthwise_conv_nt_t_s8(weight_sum_buf + block_offset,  // Pre-computed weight sums
                                                  lhs_buffer,                     // Im2col buffer (4 positions)
                                                  kernel + block_offset,          // Kernel weights for current block
                                                  input_offset,                   // Input quantization offset
                                                  active_ch,                      // Number of active channels
                                                  input_ch,                       // Total input channels
                                                  output_shift + block_offset,    // Output quantization shifts
                                                  output_mult + block_offset,     // Output quantization multipliers
                                                  output_offset,                  // Output quantization offset
                                                  output_activation_min,          // Activation minimum
                                                  output_activation_max,          // Activation maximum
                                                  kernel_size,                    // Total kernel positions
                                                  bias + block_offset,            // Bias values for current block
                                                  out);                           // Output buffer

                    /* Advance output pointer by 4 positions × input_ch channels */
                    out += (4 * input_ch);
                    
                    /* Reset batch counter for next batch */
                    buffer_count = 0;
                }
            }
        }
        /* ========================================================================
         * REMAINING BUFFER PROCESSING
         * ========================================================================
         * Handle leftover output positions that don't fill a complete batch of 4.
         * These are processed individually using MVE vectorized operations. */
        
        /* Reset im2col buffer pointer to beginning */
        lhs_buffer = (int8_t *)buffer_a;

        /* Store base output pointer for remaining positions */
        int8_t *out_base = out;
        
        /* Process each remaining output position individually */
        for (int i_buf = 0; i_buf < buffer_count; i_buf++)
        {
            /* ========================================================================
             * CHANNEL VECTORIZATION LOOP
             * ========================================================================
             * Process channels in groups of 4 using MVE vectorized operations.
             * This ensures maximum utilization of MVE capabilities. */
            
            /* Calculate number of 4-channel vector iterations needed */
            int32_t loop_count = (active_ch + 3) / 4;
            
            /* Track remaining channels to process in current output position */
            int32_t num_ch_to_process = active_ch;
            
            /* Set output pointer for current buffer position */
            out = out_base + (i_buf * input_ch);
            
            /* Process channels in groups of 4 using MVE vectors */
            for (int i_loop_cnt = 0, offset = i_ch * CH_IN_BLOCK_MVE; i_loop_cnt < loop_count;
                 num_ch_to_process -= 4, offset += 4, i_loop_cnt++)
            {
                /* ========================================================================
                 * MVE VECTORIZED MULTIPLY-ACCUMULATE
                 * ========================================================================
                 * Perform depthwise convolution using MVE vectorized operations.
                 * This is the core computational kernel for remaining positions. */
                
                /* Get pointers to current 4-channel group in im2col buffer and kernel */
                const int8_t *col_0 = lhs_buffer + (kernel_size * CH_IN_BLOCK_MVE * i_buf) + (i_loop_cnt * 4);
                const int8_t *row_0 = kernel + offset;
                
                /* Initialize accumulator vector to zero */
                int32x4_t out_0 = vdupq_n_s32(0);

                /* ========================================================================
                 * KERNEL ITERATION WITH MVE VECTORIZATION
                 * ========================================================================
                 * For each kernel position, perform vectorized multiply-accumulate
                 * operations using MVE instructions. */
                
                for (int i_ker = 0; i_ker < kernel_size; i_ker++)
                {
                    /* Load 4 kernel values into MVE vector (int8 -> int32) */
                    const int32x4_t ker_0 = vldrbq_s32(row_0);
                    
                    /* Load 4 input values into MVE vector (int8 -> int32) */
                    int32x4_t ip_0 = vldrbq_s32(col_0);
                    
                    /* Vectorized multiply-accumulate: out_0 += ip_0 * ker_0 */
                    out_0 += vmulq_s32(ip_0, ker_0);
                    
                    /* Advance pointers to next kernel position */
                    col_0 += CH_IN_BLOCK_MVE;  // Next position in im2col buffer
                    row_0 += input_ch;         // Next position in kernel
                }
                
                /* ========================================================================
                 * MVE PREDICATED OPERATIONS FOR REMAINING CHANNELS
                 * ========================================================================
                 * Use MVE predicated execution to handle partial vectors efficiently.
                 * This ensures we only process valid channels in the last iteration. */
                
                /* Create predicate mask for remaining channels */
                mve_pred16_t p = vctp32q((uint32_t)num_ch_to_process);
                
                /* Add pre-computed weight sums (predicated load) */
                out_0 += vldrwq_z_s32(&weight_sum_buf[offset], p);

                /* ========================================================================
                 * QUANTIZATION AND ACTIVATION
                 * ========================================================================
                 * Apply quantization and activation functions using MVE operations. */
                
                /* Load quantization parameters */
                const int32x4_t mult = vldrwq_s32(&output_mult[offset]);
                const int32x4_t shift = vldrwq_s32(&output_shift[offset]);

                /* Apply requantization: out = (sum * mult) >> shift */
                out_0 = arm_requantize_mve_32x4(out_0, mult, shift);
                
                /* Add output offset */
                out_0 = vaddq_n_s32(out_0, output_offset);
                
                /* Apply activation function (clamp to [min, max]) */
                out_0 = vmaxq_s32(out_0, vdupq_n_s32(output_activation_min));
                out_0 = vminq_s32(out_0, vdupq_n_s32(output_activation_max));
                
                /* Store results (predicated store - only store valid channels) */
                vstrbq_p_s32(out, out_0, p);

                /* Advance output pointer by 4 channels */
                out += 4;
            }
        }
        
        /* Reset buffer counter for next channel block */
        buffer_count = 0;

        /* ========================================================================
         * CHANNEL BLOCK ADVANCEMENT
         * ========================================================================
         * Move to next channel block and update active channel count. */
        
        /* Calculate active channels for next block */
        active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch);
        remaining_ch -= CH_IN_BLOCK_MVE;
    }

    #else // ARM_MATH_DSP
    (void)weight_sum_ctx;

    /* Run the following code in cores using DSP extension */
    int16_t *const col_buffer_start = buffer_a;
    int16_t *col_buffer = col_buffer_start;
    const int32_t *const bias_start_pos = bias;
    const int32_t *const out_mult_start_pos = output_mult;
    const int32_t *const out_shift_start_pos = output_shift;
    uint16_t row_count;
    uint16_t row_shift;

    for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
        for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;

            /* Out of bounds is only considered for the y axis as it provides a contiguous zero'ing opportunity than
               along the x axis */
            const int ker_y_start = MAX(0, -base_idx_y);
            /* Condition for kernel end dimension: (base_idx_y + ker_y_end) < input_y */
            const int ker_y_end = MIN(kernel_y, input_y - base_idx_y);

            int32_t index = 0;
            if (ker_y_start != 0)
            {
                memset(&col_buffer[index], 0, (kernel_x * input_ch) * ker_y_start * sizeof(int16_t));
                index += (kernel_x * input_ch) * ker_y_start;
            }

            for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
            {
                const int32_t idx_y = base_idx_y + i_ker_y;

                for (int i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                {
                    const int32_t idx_x = base_idx_x + i_ker_x;
                    if (idx_x < 0 || idx_x >= input_x)
                    {
                        memset(&col_buffer[index], 0, input_ch * sizeof(int16_t));
                    }
                    else
                    {
                        arm_q7_to_q15_with_offset((int8_t *)input + (idx_y * input_x + idx_x) * input_ch,
                                                  &col_buffer[index],
                                                  input_ch,
                                                  (int16_t)input_offset);
                    }
                    index += input_ch;
                }
            }

            const int diff = kernel_y - ker_y_end;
            if (diff != 0)
            {
                memset(&col_buffer[index], 0, (kernel_x * input_ch) * diff * sizeof(int16_t));
            }

            row_count = output_ch / 4;
            row_shift = 0;
            bias = bias_start_pos;
            output_mult = out_mult_start_pos;
            output_shift = out_shift_start_pos;

            while (row_count)
            {
                int32_t sum = 0;
                int32_t sum_2 = 0;
                int32_t sum_3 = 0;
                int32_t sum_4 = 0;
                if (bias)
                {
                    sum = *bias++;
                    sum_2 = *bias++;
                    sum_3 = *bias++;
                    sum_4 = *bias++;
                }

                uint16_t col_count = (kernel_x * kernel_y) / 2;
                int16_t *col_pos = col_buffer_start + row_shift;
                const int8_t *row_pos = kernel + row_shift;
                row_shift += 4;

                while (col_count)
                {
                    /* General idea is to read 4 + 4 (input, kernel) pair and re-arrange them in the right order to
                    use in a SMLAD instruction . One run of this loop produces 4 partial outputs with 8 MACs. */
                    /* Note: variable names can be improved here to align with rows and columns. */
                    int32_t ip_a1, ip_a2, ip_b1, ip_b2, op_a, op_b, op_c;
                    /* Read 4 weights */
                    ip_b1 = arm_nn_read_s8x4(row_pos);
                    ip_a1 = arm_nn_read_s8x4(row_pos + input_ch);
                    op_a = arm_nn_read_s16x2(col_pos);
                    op_b = arm_nn_read_s16x2(col_pos + input_ch);

                    ip_a2 = SXTB16(ip_b1);
                    ip_b1 = SXTB16(ROR(ip_b1, 8));

                    ip_b2 = SXTB16(ip_a1);
                    ip_a1 = SXTB16(ROR(ip_a1, 8));

                    op_c = PKHBT(op_b, op_a, 16);
                    op_a = PKHTB(op_b, op_a, 16);
                    op_b = PKHBT(ip_b2, ip_a2, 16);
                    sum = SMLAD(op_c, op_b, sum);

                    op_b = PKHBT(ip_b1, ip_a1, 16);
                    sum_2 = SMLAD(op_a, op_b, sum_2);

                    op_a = arm_nn_read_s16x2(col_pos + 2);
                    op_b = arm_nn_read_s16x2(col_pos + input_ch + 2);

                    op_c = PKHBT(op_b, op_a, 16);
                    op_a = PKHTB(op_b, op_a, 16);
                    op_b = PKHTB(ip_a2, ip_b2, 16);
                    sum_3 = SMLAD(op_c, op_b, sum_3);

                    op_b = PKHTB(ip_a1, ip_b1, 16);
                    sum_4 = SMLAD(op_a, op_b, sum_4);

                    row_pos += input_ch << 1;
                    col_pos += input_ch << 1;
                    col_count--;
                }

                col_count = (kernel_x * kernel_y) & 0x1;
                while (col_count)
                {
                    sum += row_pos[0] * col_pos[0];
                    sum_2 += row_pos[1] * col_pos[1];
                    sum_3 += row_pos[2] * col_pos[2];
                    sum_4 += row_pos[3] * col_pos[3];

                    row_pos += input_ch;
                    col_pos += input_ch;

                    col_count--;
                }
                sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                sum += output_offset;
                sum = MAX(sum, output_activation_min);
                sum = MIN(sum, output_activation_max);
                *output++ = (int8_t)sum;

                sum_2 = arm_nn_requantize(sum_2, *output_mult++, *output_shift++);
                sum_2 += output_offset;
                sum_2 = MAX(sum_2, output_activation_min);
                sum_2 = MIN(sum_2, output_activation_max);
                *output++ = (int8_t)sum_2;
                sum_3 = arm_nn_requantize(sum_3, *output_mult++, *output_shift++);
                sum_3 += output_offset;
                sum_3 = MAX(sum_3, output_activation_min);
                sum_3 = MIN(sum_3, output_activation_max);
                *output++ = (int8_t)sum_3;

                sum_4 = arm_nn_requantize(sum_4, *output_mult++, *output_shift++);
                sum_4 += output_offset;
                sum_4 = MAX(sum_4, output_activation_min);
                sum_4 = MIN(sum_4, output_activation_max);
                *output++ = (int8_t)sum_4;

                row_count--;
            }

            row_count = output_ch & 0x3;
            while (row_count)
            {
                int16_t *col_pos = col_buffer_start + row_shift;
                const int8_t *row_pos = kernel + row_shift;
                int32_t sum = 0;
                if (bias)
                {
                    sum = *bias++;
                }
                const uint16_t col_count = (kernel_x * kernel_y);
                row_shift += 1;

                for (int i = 0; i < col_count; i++)
                {
                    sum += row_pos[i * input_ch] * col_pos[i * input_ch];
                }
                sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                sum += output_offset;
                sum = MAX(sum, output_activation_min);
                sum = MIN(sum, output_activation_max);
                *output++ = (int8_t)sum;

                row_count--;
            }

            // clear counter and pointers
            col_buffer = col_buffer_start;
        }
    }
    #endif
#else
    (void)weight_sum_ctx;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    return arm_depthwise_conv_s8(ctx,
                                 dw_conv_params,
                                 quant_params,
                                 input_dims,
                                 input,
                                 filter_dims,
                                 kernel,
                                 bias_dims,
                                 bias,
                                 output_dims,
                                 output);
#endif /* ARM_MATH_MVEI | ARM_MATH_DSP */

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

#if defined(ARM_MATH_MVEI)
static inline void arm_depthwise_conv_s8_1d_valid_1out(
    const int32_t *weight_sum_base,  // base[c] = bias[c] + input_offset * sum_w[c]
    const int8_t  *lhs_col0,         // &input[(x0)*C + ch_block_offset]
    const int8_t  *rhs,              // weights (tap-major), already +ch_block_offset
    int32_t        active_ch,        // <= CH_IN_BLOCK_MVE
    int32_t        total_ch,         // C
    const int32_t *out_shift,        // already +ch_block_offset
    const int32_t *out_mult,         // already +ch_block_offset
    int32_t        out_offset,
    int32_t        act_min,
    int32_t        act_max,
    int            kernel_x,
    int8_t        *out_col)          // &output[(x0)*C + ch_block_offset]
{
    uint32_t rem = active_ch;

    for (int off = 0; off < active_ch; off += 4, rem -= 4, out_col += 4)
    {
        mve_pred16_t p = vctp32q(rem);

        int32x4_t acc  = vldrwq_z_s32(weight_sum_base + off, p);
        int32x4_t mult = vldrwq_z_s32(out_mult         + off, p);
        int32x4_t shft = vldrwq_z_s32(out_shift        + off, p);

        // loop over taps
        const int8_t *x_t = lhs_col0 + off;
        const int8_t *w_t = rhs       + off;
        for (int t = 0; t < kernel_x; ++t) {
            int32x4_t xv = vldrbq_z_s32(x_t, p);   // 4 lanes = 4 channels (s8->s32)
            int32x4_t wv = vldrbq_z_s32(w_t, p);   // matching 4 weights
            acc = vmlaq_s32(acc, xv, wv);
            x_t += total_ch;  // next tap input (x0 + t + 1)
            w_t += total_ch;  // next tap weights
        }

        acc = arm_requantize_mve_32x4(acc, mult, shft);
        acc = vaddq_n_s32(acc, out_offset);
        acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
        acc = vminq_s32(acc, vdupq_n_s32(act_max));
        vstrbq_p_s32(out_col, acc, p);
    }
}

arm_cmsis_nn_status arm_depthwise_conv_s8_1d_valid(const int32_t* weight_sum_buf,  // Pre-computed weight sums for optimization
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
        mve_pred16_t p = vctp32q(num_ch_to_process);
        // per-channel state
        int32x4_t base  = vldrwq_z_s32(&weight_sum_base[offset], p); // this is the pre-computed weight sums
        int32x4_t mult  = vldrwq_z_s32(&out_mult[offset],        p);
        int32x4_t shift  = vldrwq_z_s32(&out_shift[offset],       p);

        /* Initialize 4 output accumulators (one for each output position) */
        int32x4_t out_0 = base;  // Accumulator for output position 0
        int32x4_t out_1 = base;  // Accumulator for output position 1
        int32x4_t out_2 = base;  // Accumulator for output position 2
        int32x4_t out_3 = base;  // Accumulator for output position 3


        /* =======================================================================
        * INPUT DATA POINTER SETUP
        * ========================================================================
        * Set up pointers to access 4 input matrices and kernel weights.
        * Each input matrix represents one output position and is separated
        * by row_x_col * CH_IN_BLOCK_MVE elements in memory. */

        /* Kernel weights pointer for current 4-channel group */
        const int8_t *rhs_0 = rhs + offset;

        /* Input data pointers for 4 output positions 
        lhs should point to the beginning of the input data*/
        const int8_t *lhs_0 = lhs + 0 * total_ch + offset;                                    // Input matrix 0 (output position 0)
        const int8_t *lhs_1 = lhs + 1 * total_ch + offset;  // column x0+1
        const int8_t *lhs_2 = lhs + 2 * total_ch + offset;  // column x0+2
        const int8_t *lhs_3 = lhs + 3 * total_ch + offset;  // column x0+3
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

            const int32x4_t ker_0 = vldrbq_z_s32(rhs_0, p);

            /* ========================================================================
            * VECTORIZED MULTIPLY-ACCUMULATE FOR 4 OUTPUT POSITIONS
            * ========================================================================
            * Process all 4 output positions in parallel using MVE vectorized operations.
            * Each position performs: accumulator += input_vector * kernel_vector */

            /* Output Position 0: Load input data and multiply-accumulate */
            int32x4_t ip_0 = vldrbq_z_s32(lhs_0, p);        // Load 4 input values (int8 -> int32x4)
            out_0 += vmulq_s32(ip_0, ker_0);           // Vectorized multiply-accumulate

            /* Output Position 1: Load input data and multiply-accumulate */
            int32x4_t ip_1 = vldrbq_z_s32(lhs_1, p);        // Load 4 input values (int8 -> int32x4)
            out_1 += vmulq_s32(ip_1, ker_0);           // Vectorized multiply-accumulate

            /* Output Position 2: Load input data and multiply-accumulate */
            int32x4_t ip_2 = vldrbq_z_s32(lhs_2, p);        // Load 4 input values (int8 -> int32x4)
            out_2 += vmulq_s32(ip_2, ker_0);           // Vectorized multiply-accumulate

            /* Output Position 3: Load input data and multiply-accumulate */
            int32x4_t ip_3 = vldrbq_z_s32(lhs_3, p);        // Load 4 input values (int8 -> int32x4)
            out_3 += vmulq_s32(ip_3, ker_0);           // Vectorized multiply-accumulate

            /* ========================================================================
            * POINTER ADVANCEMENT FOR NEXT KERNEL POSITION
            * ========================================================================
            * Advance input pointers to next kernel position for all 4 output positions.
            * Advance kernel pointer to next kernel position. */

            /* Advance input data pointers to next kernel position */
            lhs_0 += total_ch;  // Next kernel position in input matrix 0
            lhs_1 += total_ch;  // Next kernel position in input matrix 1
            lhs_2 += total_ch;  // Next kernel position in input matrix 2
            lhs_3 += total_ch;  // Next kernel position in input matrix 3

            /* Advance kernel pointer to next kernel position */
            rhs_0 += total_ch;         // Next kernel position in weight matrix
        }

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

    return ARM_CMSIS_NN_SUCCESS;
}
#endif

//this wrapper requires 1D valid kernel
arm_cmsis_nn_status arm_depthwise_conv_s8_opt_1d_valid_wrapper(const cmsis_nn_context *ctx,
    const cmsis_nn_context *weight_sum_ctx,
    const cmsis_nn_dw_conv_params *dw_conv_params,
    const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims,
    const int8_t *input,
    const cmsis_nn_dims *filter_dims,
    const int8_t *kernel,
    const cmsis_nn_dims *bias_dims,
    const int32_t *bias,
    const cmsis_nn_dims *output_dims,
    int8_t *output)
{
    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;

    /* Check depth multiplier is 1 */
    if (input_ch != output_ch)
    {
    return ARM_CMSIS_NN_ARG_ERROR;
    }

#ifdef ARM_MATH_MVEI
    (void)bias_dims;

    const int32_t input_x  = input_dims->w;
    const int32_t input_y  = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x    = dw_conv_params->padding.w;
    const int32_t pad_y    = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t *output_shift = quant_params->shift;
    const int32_t *output_mult  = quant_params->multiplier;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t out_offset = dw_conv_params->output_offset;
    const int32_t in_offset  = dw_conv_params->input_offset;
    const int32_t act_min    = dw_conv_params->activation.min;
    const int32_t act_max    = dw_conv_params->activation.max;

    // -------- SLIDING-WINDOW FAST PATH GUARD --------
    const bool is_1d_valid =
        (input_dims->n == 1) &&
        (input_y == 1) && (output_y == 1) &&
        (kernel_y == 1) &&
        (pad_x == 0) && (pad_y == 0) &&
        (stride_x == 1) && (stride_y == 1) &&
        (dw_conv_params->dilation.w == 1) && (dw_conv_params->dilation.h == 1) &&
        (dw_conv_params->ch_mult == 1);

    if (is_1d_valid)
    {
        // We will vectorize across channels and compute 4 outputs per call, no im2col.
        // Requirements:
        // - weight_sum_ctx->buf must be precomputed by arm_depthwise_convolve_weight_sum() in your init()
        int32_t *weight_sum_buf = (int32_t *)weight_sum_ctx->buf;
        if (weight_sum_buf == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        const int32_t C = input_ch;          // channels
        const int32_t Wout = output_x;       // width outputs

        // Channel blocking identical to the im2col path (fixed pitch expectation inside microkernels)
        const int32_t ch_loop = (C + (CH_IN_BLOCK_MVE - 1)) / CH_IN_BLOCK_MVE;
        int32_t remaining_ch = C;

        for (int i_ch = 0; i_ch < ch_loop; i_ch++)
        {
            const int32_t block_offset = i_ch * CH_IN_BLOCK_MVE; // 0
            const int32_t active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch); // 40
            remaining_ch -= CH_IN_BLOCK_MVE;

            // Slide across width in tiles of 4 outputs
            int x0 = 0;
            for (; x0 + 3 < Wout; x0 += 4)
            {
                // Base pointers into input/output at column x0 and channel block
                const int8_t *lhs_base = input  + (x0 * C) + block_offset;
                int8_t       *out_base = output + (x0 * C) + block_offset;

                // Call your 4-output sliding-window microkernel
                arm_cmsis_nn_status status = arm_depthwise_conv_s8_1d_valid(
                    weight_sum_buf + block_offset,   // base[c]
                    lhs_base,                         // &input[(x0)*C + block_offset]
                    kernel + block_offset,            // weights + channel block
                    in_offset,                        // unused inside microkernel (kept for signature parity)
                    active_ch,                        // <= CH_IN_BLOCK_MVE
                    C,                                // total_ch = C
                    output_shift + block_offset,      // per-channel shift (block-local base)
                    output_mult  + block_offset,      // per-channel mult  (block-local base)
                    out_offset,
                    act_min,
                    act_max,
                    kernel_x,
                    NULL,             // unused (compensation folded in base[])
                    out_base);                        // &output[(x0)*C + block_offset]

                if (status != ARM_CMSIS_NN_SUCCESS) return status;
            }

            // Tail: 0..3 leftover outputs → use 1-output sliding microkernel
            for (; x0 < Wout; x0++)
            {
                const int8_t *lhs_col0 = input  + (x0 * C) + block_offset;
                int8_t       *out_col  = output + (x0 * C) + block_offset;

                arm_depthwise_conv_s8_1d_valid_1out(
                    weight_sum_buf + block_offset,
                    lhs_col0,
                    kernel + block_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    out_col);
            }
        }

        // Fast path handled everything
        return ARM_CMSIS_NN_SUCCESS;
    }
#else
    (void)weight_sum_ctx;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    return arm_depthwise_conv_s8(ctx,
                                dw_conv_params,
                                quant_params,
                                input_dims,
                                input,
                                filter_dims,
                                kernel,
                                bias_dims,
                                bias,
                                output_dims,
                                output);
#endif  // ARM_MATH_MVEI
}

/**
 * @} end of NNConv group
 */
