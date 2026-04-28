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
    // Unpack dims for convenience
    const int32_t out_n = output_dims->n, out_h = output_dims->h;
    const int32_t out_w = output_dims->w, out_c = output_dims->c;

    const int32_t in_n = input_dims->n, in_h = input_dims->h;
    const int32_t in_w = input_dims->w, in_c = input_dims->c;

    const int32_t alpha_n = alpha_dims->n, alpha_h = alpha_dims->h;
    const int32_t alpha_w = alpha_dims->w, alpha_c = alpha_dims->c;

    // PReLU is elementwise (input_dims == output_dims)
    if (in_n != out_n || in_h != out_h || in_w != out_w || in_c != out_c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Total element‐counts
    int32_t flat1_total = in_n * in_h * in_w * in_c;
    int32_t flat2_total = alpha_n * alpha_h * alpha_w * alpha_c;

    // 1) No broadcast at all? (identical sizes)
    if (!arm_check_broadcast_required(input_dims, alpha_dims))
    {
        return arm_elementwise_prelu_s8(input,
                                        alpha,
                                        input_offset,
                                        alpha_offset,
                                        output_offset,
                                        output_multiplier_identity,
                                        output_shift_identity,
                                        output_multiplier_alpha,
                                        output_shift_alpha,
                                        output,
                                        flat1_total);
    }

    // 2) Whole‐tensor scalar?
    if (flat1_total == 1)
    {
        return arm_prelu_scalar_s8(input,
                                   alpha,
                                   true,
                                   input_offset,
                                   alpha_offset,
                                   output_offset,
                                   output_multiplier_identity,
                                   output_shift_identity,
                                   output_multiplier_alpha,
                                   output_shift_alpha,
                                   output,
                                   flat2_total);
    }
    // Scalar alpha
    if (flat2_total == 1)
    {
        return arm_prelu_scalar_s8(alpha,
                                   input,
                                   false,
                                   input_offset,
                                   alpha_offset,
                                   output_offset,
                                   output_multiplier_identity,
                                   output_shift_identity,
                                   output_multiplier_alpha,
                                   output_shift_alpha,
                                   output,
                                   flat1_total);
    }

    // 3) Full N/H/W/C broadcast sweep
    // Compute “rewind” and “advance” offsets same as arm_maximum_s8
    const int32_t wd1 = (in_w >= alpha_w) ? 0 : in_c;
    const int32_t wd2 = (alpha_w >= in_w) ? 0 : alpha_c;
    const int32_t hd1 = (in_h >= alpha_h) ? wd1 : -in_w * (in_c - wd1);
    const int32_t hd2 = (alpha_h >= in_h) ? wd2 : -alpha_w * (alpha_c - wd2);
    const int32_t bd1 = (in_n >= alpha_n) ? in_h * in_w * in_c : 0;
    const int32_t bd2 = (alpha_n >= in_n) ? alpha_h * alpha_w * alpha_c : 0;

    for (int b = 0; b < out_n; b++)
    {
        // Per‐batch pointers
        const int8_t *p1 = input;
        const int8_t *p2 = alpha;
        // Recompute per‐batch flat sizes
        flat1_total = in_h * in_w * in_c;
        flat2_total = alpha_h * alpha_w * alpha_c;

        // Batch‐level no‐broadcast?
        if (in_h == alpha_h && in_w == alpha_w && in_c == alpha_c)
        {
            arm_elementwise_prelu_s8(p1,
                                     p2,
                                     input_offset,
                                     alpha_offset,
                                     output_offset,
                                     output_multiplier_identity,
                                     output_shift_identity,
                                     output_multiplier_alpha,
                                     output_shift_alpha,
                                     output,
                                     flat1_total);
            output += flat1_total;
        }
        else
        {
            // Sweep height
            flat1_total = in_w * in_c;
            flat2_total = alpha_w * alpha_c;
            for (int h = 0; h < out_h; h++)
            {
                // A) Row‐level no‐broadcast
                if (in_w == alpha_w && in_c == alpha_c)
                {
                    arm_elementwise_prelu_s8(p1,
                                             p2,
                                             input_offset,
                                             alpha_offset,
                                             output_offset,
                                             output_multiplier_identity,
                                             output_shift_identity,
                                             output_multiplier_alpha,
                                             output_shift_alpha,
                                             output,
                                             flat1_total);
                    p1 += flat1_total;
                    p2 += flat1_total;
                    output += flat1_total;
                }
                // B) scalar‐broadcast on input1
                else if (flat1_total == 1)
                {
                    arm_prelu_scalar_s8(p1,
                                        p2,
                                        true,
                                        input_offset,
                                        alpha_offset,
                                        output_offset,
                                        output_multiplier_identity,
                                        output_shift_identity,
                                        output_multiplier_alpha,
                                        output_shift_alpha,
                                        output,
                                        flat2_total);
                    p2 += flat2_total;
                    output += flat2_total;
                }
                // C) scalar‐broadcast on input2
                else if (flat2_total == 1)
                {
                    arm_prelu_scalar_s8(p2,
                                        p1,
                                        false,
                                        input_offset,
                                        alpha_offset,
                                        output_offset,
                                        output_multiplier_identity,
                                        output_shift_identity,
                                        output_multiplier_alpha,
                                        output_shift_alpha,
                                        output,
                                        flat1_total);
                    p1 += flat1_total;
                    output += flat1_total;
                }
                // D) per‐pixel broadcast sweep
                else
                {
                    for (int w = 0; w < out_w; w++)
                    {
                        // 1) per‐pixel no‐broadcast
                        if (in_c == alpha_c)
                        {
                            arm_elementwise_prelu_s8(p1,
                                                     p2,
                                                     input_offset,
                                                     alpha_offset,
                                                     output_offset,
                                                     output_multiplier_identity,
                                                     output_shift_identity,
                                                     output_multiplier_alpha,
                                                     output_shift_alpha,
                                                     output,
                                                     in_c);
                            p1 += in_c;
                            p2 += in_c;
                            output += in_c;
                        }
                        // 2) scalar‐broadcast1
                        else if (in_c == 1)
                        {
                            arm_prelu_scalar_s8(p1,
                                                p2,
                                                true,
                                                input_offset,
                                                alpha_offset,
                                                output_offset,
                                                output_multiplier_identity,
                                                output_shift_identity,
                                                output_multiplier_alpha,
                                                output_shift_alpha,
                                                output,
                                                alpha_c);
                            p1++;
                            p2 += alpha_c;
                            output += alpha_c;
                        }
                        // 3) scalar‐broadcast2
                        else // alpha_c == 1
                        {
                            arm_prelu_scalar_s8(p2,
                                                p1,
                                                false,
                                                input_offset,
                                                alpha_offset,
                                                output_offset,
                                                output_multiplier_identity,
                                                output_shift_identity,
                                                output_multiplier_alpha,
                                                output_shift_alpha,
                                                output,
                                                in_c);
                            p1 += in_c;
                            p2++;
                            output += in_c;
                        }
                        // rewind for next width
                        p1 -= wd1;
                        p2 -= wd2;
                    }
                }
                // advance row
                p1 += hd1;
                p2 += hd2;
            }
        }

        // advance batch
        input += bd1;
        alpha += bd2;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
