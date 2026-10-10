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
 * Title:        arm_transpose_s8.c
 * Description:  Transpose a s8 vector
 *
 * $Date:        30 October 2024
 * $Revision:    V.1.0.1
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
 * @addtogroup Transpose
 * @{
 */

/*
 * Transpose function specialized for int16 data in a NHCW (batch, height, channels, width)
 * arrangement. This version uses MVE intrinsics when available.
 */
static arm_cmsis_nn_status arm_transpose_s16_nhcw(const int16_t *input,
                                                  int16_t *const output,
                                                  const cmsis_nn_dims *const input_dims,
                                                  const int32_t *const in_strides,
                                                  const int32_t *const out_strides)
{
    const int32_t n = input_dims->n;
    const int32_t h = input_dims->h;
    const int32_t w = input_dims->w;
    const int32_t c = input_dims->c;

    const int16_t *input_n = input;
    int16_t *output_n = output;

    /* Here we interpret the source as a 2D matrix of shape:
       src_rows = w (number of elements per row)
       src_cols = c (number of columns)
       (this arrangement matches the expected layout for the optimized transpose)
    */
    const int32_t src_rows = w;
    const int32_t src_cols = c;

#if defined(ARM_MATH_MVEI)
    uint16x8_t vec_offsets;
    uint16x8_t vec_input;

    /* Create a vector of indices [0, 1, 2, ..., 7] and multiply by src_cols */
    vec_offsets = vidupq_u16((uint32_t)0, 1);
    vec_offsets = vec_offsets * (uint16_t)src_cols;
#endif

    for (int32_t i = 0; i < n; i++)
    {
        const int16_t *input_h = input_n;
        int16_t *output_h = output_n;

        for (int32_t y = 0; y < h; y++)
        {

#if defined(ARM_MATH_MVEI)
            /* For the current row, work on each column of the source matrix */
            const uint16_t *input_c = (const uint16_t *)input_h;
            uint16_t *output_c = (uint16_t *)output_h;

            for (int32_t z = 0; z < src_cols; z++)
            {
                uint16_t const *input_w = (uint16_t const *)input_c;
                uint16_t *output_w = (uint16_t *)output_c;

                int32_t block_count = src_rows;
                while (block_count > 0)
                {
                    mve_pred16_t p = vctp16q(block_count);
                    /* Gather 8 int16 values from input_w using the computed byte offsets */
                    vec_input = vldrhq_gather_shifted_offset_z_u16(input_w, vec_offsets, p);
                    /* Store the gathered vector to output_w with predication */
                    vstrhq_p_u16(output_w, vec_input, p);

                    /* Advance the input pointer by 8 rows. Since each row is src_cols int16 values,
                       we add (src_cols * 8) elements. The output pointer is advanced by 8 elements. */
                    input_w = input_w + src_cols * 8;
                    output_w += 8;
                    block_count -= 8;
                }

                input_c++;            /* Next column */
                output_c += src_rows; /* Advance output pointer by the number of rows */
            }
#else
            const uint16_t *input_w = (const uint16_t *)input_h;
            uint16_t *output_w = (uint16_t *)output_h;

            for (int32_t src_row_i = 0; src_row_i < src_rows; src_row_i++)
            {
                output_w = (uint16_t *)output_h + src_row_i;

                for (int32_t x = 0; x < src_cols; x++)
                {
                    *output_w = *input_w++;
                    output_w += src_rows;
                }
            }
#endif
            input_h += in_strides[1];
            output_h += out_strides[1];
        }
        input_n += in_strides[0];
        output_n += out_strides[0];
    }

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_transpose_s16_default(const int16_t *input,
                                                     int16_t *const output,
                                                     const cmsis_nn_dims *const input_dims,
                                                     const int32_t *const in_strides,
                                                     const int32_t *const out_strides)
{
    const int32_t n = input_dims->n;
    const int32_t h = input_dims->h;
    const int32_t w = input_dims->w;
    const int32_t c = input_dims->c;

    for (int32_t i = 0; i < n; i++)
    {
        for (int32_t y = 0; y < h; y++)
        {
            for (int32_t x = 0; x < w; x++)
            {
                for (int32_t z = 0; z < c; z++)
                {
                    const int32_t from_index =
                        i * in_strides[0] + y * in_strides[1] + x * in_strides[2] + z * in_strides[3];

                    const int32_t to_index =
                        i * out_strides[0] + y * out_strides[1] + x * out_strides[2] + z * out_strides[3];

                    output[to_index] = input[from_index];
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * Loop bounds come from input_dims but the output strides come from output_dims, so an output_dims that is not
 * input_dims permuted by perm lets to_index run past the end of the output buffer. An extent below 1 passes
 * that cross-check when both sides carry it and then wraps the unsigned copy length, so the extents are
 * rejected before any of them is used in arithmetic.
 * see AmbiqAI/ns-cmsis-nn#443
 */
static arm_cmsis_nn_status arm_transpose_s16_check_dims(const cmsis_nn_dims *const input_dims,
                                                        const cmsis_nn_dims *const output_dims,
                                                        const uint32_t *const perm,
                                                        const int32_t num_dims)
{
    if (num_dims < 1 || num_dims > 4)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t in_dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t out_dims[4] = {output_dims->n, output_dims->h, output_dims->w, output_dims->c};
    uint32_t axes_seen = 0;

    for (int32_t i = 0; i < num_dims; i++)
    {
        if (in_dims[i] < 1 || out_dims[i] < 1)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    for (int32_t i = 0; i < num_dims; i++)
    {
        const uint32_t axis = perm[i];

        if (axis >= (uint32_t)num_dims || (axes_seen & (1U << axis)) != 0U)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        axes_seen |= 1U << axis;

        if (out_dims[i] != in_dims[axis])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * Basic s16 transpose function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_transpose_s16(const int16_t *input,
                                      int16_t *const output,
                                      const cmsis_nn_dims *const input_dims,
                                      const cmsis_nn_dims *const output_dims,
                                      const cmsis_nn_transpose_params *const transpose_params)
{
    const int32_t num_dims = transpose_params->num_dims;

    /* The stride products below are signed, so the extents are validated before any arithmetic
     * derives from them. see AmbiqAI/ns-cmsis-nn#443 */
    if (arm_transpose_s16_check_dims(input_dims, output_dims, transpose_params->permutations, num_dims) !=
        ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Leading axes of extent 1 bring every rank to four, so one rule picks the path for all of them.
     * see AmbiqAI/ns-cmsis-nn#757 */
    const int32_t in_dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t pad = 4 - num_dims;
    int32_t dims[4] = {1, 1, 1, 1};
    uint32_t perm[4] = {0, 1, 2, 3};

    for (int32_t i = 0; i < num_dims; i++)
    {
        dims[pad + i] = in_dims[i];
        perm[pad + i] = transpose_params->permutations[i] + (uint32_t)pad;
    }

    if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2)
    {
        arm_memcpy_s16(output, input, (uint32_t)(dims[0] * dims[1] * dims[2] * dims[3]));

        return ARM_CMSIS_NN_SUCCESS;
    }

    const cmsis_nn_dims padded_dims = {dims[0], dims[1], dims[2], dims[3]};
    const int32_t in_strides[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    int32_t out_strides[4];

    out_strides[perm[3]] = 1;
    out_strides[perm[2]] = dims[perm[3]];
    out_strides[perm[1]] = dims[perm[3]] * dims[perm[2]];
    out_strides[perm[0]] = dims[perm[3]] * dims[perm[2]] * dims[perm[1]];

    /* The gather path holds the offsets of its 8 lanes, up to 7 * cols, in 16 bits. */
#if defined(ARM_MATH_MVEI)
    const int32_t max_swap_cols = UINT16_MAX / 7;
#else
    const int32_t max_swap_cols = INT32_MAX;
#endif

    /* Not the identity, so this swaps the last two axes. */
    if (perm[0] == 0 && perm[1] == 1 && dims[3] <= max_swap_cols)
    {
        return arm_transpose_s16_nhcw(input, output, &padded_dims, in_strides, out_strides);
    }

    return arm_transpose_s16_default(input, output, &padded_dims, in_strides, out_strides);
}

/**
 * @} end of Transpose group
 */
