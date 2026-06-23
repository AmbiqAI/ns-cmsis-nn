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
 * Title:        arm_prelu_s16.c
 * Description:  Parametric ReLU function for int16_t data type
 *
 * $Date:        23 June 2026
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
 * PReLU activation function for int16_t data type.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_prelu_s16(const cmsis_nn_dims *input_dims,
                                  const int16_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const int16_t *alpha,
                                  const int32_t input_offset,
                                  const int32_t alpha_offset,
                                  const int32_t output_offset,
                                  const int32_t output_multiplier_identity,
                                  const int32_t output_shift_identity,
                                  const int32_t output_multiplier_alpha,
                                  const int32_t output_shift_alpha,
                                  const cmsis_nn_dims *output_dims,
                                  int16_t *output)
{
    const int32_t out_n = output_dims->n, out_h = output_dims->h;
    const int32_t out_w = output_dims->w, out_c = output_dims->c;

    const int32_t in_n = input_dims->n, in_h = input_dims->h;
    const int32_t in_w = input_dims->w, in_c = input_dims->c;

    const int32_t alpha_n = alpha_dims->n, alpha_h = alpha_dims->h;
    const int32_t alpha_w = alpha_dims->w, alpha_c = alpha_dims->c;

    if (in_n != out_n || in_h != out_h || in_w != out_w || in_c != out_c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t flat1_total = in_n * in_h * in_w * in_c;
    int32_t flat2_total = alpha_n * alpha_h * alpha_w * alpha_c;

    if (!arm_check_broadcast_required(input_dims, alpha_dims))
    {
        return arm_elementwise_prelu_s16(input,
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

    if (flat1_total == 1)
    {
        return arm_prelu_scalar_s16(input,
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

    if (flat2_total == 1)
    {
        return arm_prelu_scalar_s16(alpha,
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

    const int32_t wd1 = (in_w >= alpha_w) ? 0 : in_c;
    const int32_t wd2 = (alpha_w >= in_w) ? 0 : alpha_c;
    const int32_t hd1 = (in_h >= alpha_h) ? wd1 : -in_w * (in_c - wd1);
    const int32_t hd2 = (alpha_h >= in_h) ? wd2 : -alpha_w * (alpha_c - wd2);
    const int32_t bd1 = (in_n >= alpha_n) ? in_h * in_w * in_c : 0;
    const int32_t bd2 = (alpha_n >= in_n) ? alpha_h * alpha_w * alpha_c : 0;

    for (int b = 0; b < out_n; b++)
    {
        const int16_t *p1 = input;
        const int16_t *p2 = alpha;
        flat1_total = in_h * in_w * in_c;
        flat2_total = alpha_h * alpha_w * alpha_c;

        if (in_h == alpha_h && in_w == alpha_w && in_c == alpha_c)
        {
            arm_elementwise_prelu_s16(p1,
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
            flat1_total = in_w * in_c;
            flat2_total = alpha_w * alpha_c;
            for (int h = 0; h < out_h; h++)
            {
                if (in_w == alpha_w && in_c == alpha_c)
                {
                    arm_elementwise_prelu_s16(p1,
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
                else if (flat1_total == 1)
                {
                    arm_prelu_scalar_s16(p1,
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
                else if (flat2_total == 1)
                {
                    arm_prelu_scalar_s16(p2,
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
                else
                {
                    for (int w = 0; w < out_w; w++)
                    {
                        if (in_c == alpha_c)
                        {
                            arm_elementwise_prelu_s16(p1,
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
                        else if (in_c == 1)
                        {
                            arm_prelu_scalar_s16(p1,
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
                        else
                        {
                            arm_prelu_scalar_s16(p2,
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
                        p1 -= wd1;
                        p2 -= wd2;
                    }
                }
                p1 += hd1;
                p2 += hd2;
            }
        }

        input += bd1;
        alpha += bd2;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */