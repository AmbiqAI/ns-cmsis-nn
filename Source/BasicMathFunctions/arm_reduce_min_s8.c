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
 * Title:        arm_reduce_min_s8
 * Description:  Reduce min operator for quantized s8 tensors
 *
 * $Date:        23 June 2025
 * $Revision:    V.1.0.0
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
 * @addtogroup Reduction
 * @{
 */


/**
 * @brief Generic fallback reduce min operator for quantized s8 tensors.
 *
 * @param[in]  input_data     Pointer to input tensor
 * @param[in]  input_dims     Input tensor dimensions (4D NHWC)
 * @param[in]  axis_dims      4D binary axis mask (1 = reduce)
 * @param[out] output_data    Pointer to output buffer
 * @param[in]  output_dims    Output tensor dimensions
 *
 * @return     ARM_CMSIS_NN_SUCCESS on success
 */
arm_cmsis_nn_status arm_reduce_min_generic_s8(const int8_t *input_data,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *axis_dims,
                                              int8_t *output_data,
                                              const cmsis_nn_dims *output_dims)
{

    const int N = input_dims->n;
    const int H = input_dims->h;
    const int W = input_dims->w;
    const int C = input_dims->c;

    const int out_N = output_dims->n;
    const int out_H = output_dims->h;
    const int out_W = output_dims->w;
    const int out_C = output_dims->c;

    const int N_limit = axis_dims->n ? N : 1;
    const int H_limit = axis_dims->h ? H : 1;
    const int W_limit = axis_dims->w ? W : 1;
    const int C_limit = axis_dims->c ? C : 1;

    for (int n = 0; n < out_N; ++n)
    for (int h = 0; h < out_H; ++h)
    for (int w = 0; w < out_W; ++w)
    for (int c = 0; c < out_C; ++c)
    {
        int8_t min_val = INT8_MAX;

        for (int ni = 0; ni < N_limit; ++ni)
        for (int hi = 0; hi < H_limit; ++hi)
        for (int wi = 0; wi < W_limit; ++wi)
        for (int ci = 0; ci < C_limit; ++ci)
        {
            int idx_n = axis_dims->n ? ni : n;
            int idx_h = axis_dims->h ? hi : h;
            int idx_w = axis_dims->w ? wi : w;
            int idx_c = axis_dims->c ? ci : c;

            int flat_index = ((idx_n * H + idx_h) * W + idx_w) * C + idx_c;
            int8_t v = input_data[flat_index];
            if (v < min_val) min_val = v;
        }

        int out_index = ((n * out_H + h) * out_W + w) * out_C + c;
        output_data[out_index] = min_val;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @brief Flatten last dims reduce min operator for quantized s8 tensors.
 *
 * @param[in]  input_data     Pointer to input tensor
 * @param[out] output_data    Pointer to output buffer
 * @param[in]  outer_size     Size of the outer dimension (number of blocks)
 * @param[in]  inner_size     Size of the inner dimension (number of elements in each block)
 *
 * @return     ARM_CMSIS_NN_SUCCESS on success
 */
arm_cmsis_nn_status arm_reduce_min_flatten_last_dims_s8(const int8_t *input_data,
                                                        int8_t *output_data,
                                                        int32_t outer_size,
                                                        int32_t inner_size)
{
#if defined(ARM_MATH_MVEI)

    const int8x16_t vmax_s8 = vdupq_n_s8(INT8_MAX);
    for (int i = 0; i < outer_size; ++i)
    {
        // pointer to this block
        const int8_t *p = input_data + i * inner_size;

        // Vector-wide running min
        int8x16_t vmin = vmax_s8;

        int32_t j = 0;
        // Process 16 lanes at a time
        for (; j <= inner_size - 16; j += 16) {
            int8x16_t v = vld1q_s8(p + j);
            vmin = vminq_s8(vmin, v);
        }
        // Horizontal reduce-min over the vector
        int8_t min_val = INT8_MAX;
        min_val = vminvq_s8(min_val, vmin);

        // Tail
        for (; j < inner_size; ++j) {
            min_val = MIN(min_val, p[j]);
        }

        output_data[i] = min_val;
    }

    return ARM_CMSIS_NN_SUCCESS;

#elif defined(ARM_MATH_DSP)

    const int8_t *in_ptr = input_data;
    int8_t *out_ptr = output_data;

    for (int i = 0; i < outer_size; ++i)
    {
        // Start with the smallest possible int8
        int8_t min_val = INT8_MAX;

        int j = 0;
        // Process 4 at a time
        for (; j <= inner_size - 4; j += 4)
        {
            int32_t packed0, packed1;
            in_ptr = read_and_pad(in_ptr, &packed0, &packed1);
            // packed0 holds two sign-extended bytes in its low & high half-words
            int16_t lane0 = (int16_t)(packed0 & 0xFFFF);
            int16_t lane1 = (int16_t)(packed0 >> 16);
            int16_t m = min_val;
            m = lane0 < m ? lane0 : m;
            m = lane1 < m ? lane1 : m;

            // packed1 holds the next two bytes
            int16_t lane2 = (int16_t)(packed1 & 0xFFFF);
            int16_t lane3 = (int16_t)(packed1 >> 16);
            m = lane2 < m ? lane2 : m;
            m = lane3 < m ? lane3 : m;

            min_val = m;
        }

        // Handle any remaining “tail” elements
        for (; j < inner_size; ++j)
        {
            int8_t v = *in_ptr++;
            if (v < min_val)
              min_val = v;
        }

        // Store result
        *out_ptr++ = min_val;
    }

#else

    const int8_t *in_ptr = input_data;
    int8_t *out_ptr = output_data;
    for (int i = 0; i < outer_size; ++i)
    {
        int8_t min_val = *in_ptr++;
        for (int j = 1; j < inner_size; ++j)
        {
            int8_t v = *in_ptr++;
            if (v < min_val) min_val = v;
        }
        *out_ptr++ = min_val;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}


/**
 * @brief Spatial reduce min operator for quantized s8 tensors.
 *
 * @param[in]  input_data     Pointer to input tensor
 * @param[in]  input_dims     Input tensor dimensions (4D NHWC)
 * @param[in]  axis_dims      4D binary axis mask (1 = reduce)
 * @param[out] output_data    Pointer to output buffer
 * @param[in]  output_dims    Output tensor dimensions
 *
 * @return     ARM_CMSIS_NN_SUCCESS on success
 */
arm_cmsis_nn_status
arm_reduce_min_spatial_s8(const int8_t *input_data,
                          const cmsis_nn_dims *input_dims,
                          const cmsis_nn_dims *axis_dims,
                          int8_t *output_data,
                          const cmsis_nn_dims *output_dims)
{

#if defined(ARM_MATH_MVEI)

    (void)axis_dims;
    (void)output_dims;

    const int32_t N       = input_dims->n;
    const int32_t H       = input_dims->h;
    const int32_t W       = input_dims->w;
    const int32_t C       = input_dims->c;
    const int32_t spatial = H * W;

    for (int n = 0; n < N; ++n)
    {
        const int8_t *in_ptr  = input_data  + n * spatial * C;
        int8_t *out_ptr = output_data + n * C;

        int c = 0;
        for (; c <= C - 16; c += 16)
        {

            // Initialize min vector to INT8_MAX
            int8x16_t minv = vdupq_n_s8(INT8_MAX);

            // Walk over each spatial element, take pairwise min
            for (int i = 0; i < spatial; ++i)
            {
                int8x16_t v = vld1q_s8(in_ptr + i * C + c);
                minv = vminq_s8(minv, v);
            }
            // Store out the 16‐lane min
            vst1q_s8(out_ptr + c, minv);
        }

        // Scalar tail for remaining channels
        for (; c < C; ++c)
        {
            int8_t m = INT8_MAX;
            for (int i = 0; i < spatial; ++i)
            {
                int8_t v = in_ptr[i * C + c];
                if (v < m) m = v;
            }
            out_ptr[c] = m;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;

#elif defined(ARM_MATH_DSP)

    (void)axis_dims;
    (void)output_dims;

    const int32_t N       = input_dims->n;
    const int32_t H       = input_dims->h;
    const int32_t W       = input_dims->w;
    const int32_t C       = input_dims->c;
    const int32_t spatial = H * W;

    for (int n = 0; n < N; ++n) {
        const int8_t *base = input_data + n * spatial * C;
        int8_t *out        = output_data + n * C;

        for (int c = 0; c < C; ++c) {
            int s = 0;
            // Two 16-bit lanes per reg, init both to INT8_MAX
            int32_t min_pair0 = PKHBT(INT8_MAX, INT8_MAX, 16);
            int32_t min_pair1 = min_pair0;

            // Process in blocks of 4 spatial elements
            for (; s <= spatial - 4; s += 4) {
                int8_t v0 = base[(s + 0) * C + c];
                int8_t v1 = base[(s + 1) * C + c];
                int8_t v2 = base[(s + 2) * C + c];
                int8_t v3 = base[(s + 3) * C + c];

                // PACK WITH UNSIGNED CASTS to avoid sign-extension bleeding
                uint32_t op0 = (uint32_t)(uint16_t)v0
                             | ((uint32_t)(uint16_t)v1 << 16);
                uint32_t op1 = (uint32_t)(uint16_t)v2
                             | ((uint32_t)(uint16_t)v3 << 16);

                // lane‐wise min on op0 → min_pair0
                {
                  int16_t a0 = (int16_t)(op0        & 0xFFFF);
                  int16_t a1 = (int16_t)(op0 >> 16);
                  int16_t m0 = (int16_t)(min_pair0 & 0xFFFF);
                  int16_t m1 = (int16_t)(min_pair0 >> 16);
                  m0 = MIN(m0, a0);
                  m1 = MIN(m1, a1);
                  min_pair0 = PKHBT(m0, m1, 16);
                }
                // lane‐wise min on op1 → min_pair1
                {
                  int16_t a0 = (int16_t)(op1        & 0xFFFF);
                  int16_t a1 = (int16_t)(op1 >> 16);
                  int16_t m0 = (int16_t)(min_pair1 & 0xFFFF);
                  int16_t m1 = (int16_t)(min_pair1 >> 16);
                  m0 = MIN(m0, a0);
                  m1 = MIN(m1, a1);
                  min_pair1 = PKHBT(m0, m1, 16);
                }
            }

            // Collapse our two pairs into one scalar min
            int8_t m = (int8_t)( (int16_t)(min_pair0        & 0xFFFF) );
            int8_t t = (int8_t)( (int16_t)((min_pair0 >> 16) & 0xFFFF) );
            m = MIN(m, t);
            t = (int8_t)( (int16_t)(min_pair1        & 0xFFFF) );
            m = MIN(m, t);
            t = (int8_t)( (int16_t)((min_pair1 >> 16) & 0xFFFF) );
            m = MIN(m, t);

            // Handle any leftover “tail” elements
            for (; s < spatial; ++s) {
              int8_t v = base[s * C + c];
              m = MIN(m, v);
            }
            out[c] = m;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;

#else

    return arm_reduce_min_generic_s8(input_data,
                                     input_dims,
                                     axis_dims,
                                     output_data,
                                     output_dims);

#endif

}

/*
 * s8 reduce min over the specified axis
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_reduce_min_s8(const int8_t *input_data,
                                      const cmsis_nn_dims *input_dims,
                                      const cmsis_nn_dims *axis_dims,
                                      int8_t *output_data,
                                      const cmsis_nn_dims *output_dims)
{

    // Validate input
    if (!input_data || !input_dims || !axis_dims || !output_data || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t in_dims[4] = { input_dims->n, input_dims->h, input_dims->w, input_dims->c };
    int32_t axis_arr[4] = { axis_dims->n ? 1 : 0,
                            axis_dims->h ? 1 : 0,
                            axis_dims->w ? 1 : 0,
                            axis_dims->c ? 1 : 0 };

    int32_t suffix_start = arm_reduce_get_flatten_suffix_start_from_arrays(in_dims, axis_arr);

    if (suffix_start >= 0)
    {
        // compute outer/inner
        int32_t outer_size = 1, inner_size = 1;

        for (int32_t d = 0; d < suffix_start; ++d) outer_size *= in_dims[d];
        for (int32_t d = suffix_start; d < 4; ++d) inner_size *= in_dims[d];

        return arm_reduce_min_flatten_last_dims_s8(input_data,
                                                   output_data,
                                                   outer_size,
                                                   inner_size);
    }

    // Check for spatial reduction axis=[H,W]
    if (!axis_dims->n && axis_dims->h && axis_dims->w && !axis_dims->c)
    {
        return arm_reduce_min_spatial_s8(input_data,
                                         input_dims,
                                         axis_dims,
                                         output_data,
                                         output_dims);
    }

    // Fallback to general-purpose scalar implementation
    return arm_reduce_min_generic_s8(input_data,
                                     input_dims,
                                     axis_dims,
                                     output_data,
                                     output_dims);

}

/**
 * @} end of Doxygen group
 */
