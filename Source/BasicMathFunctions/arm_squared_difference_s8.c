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

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

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
    const int32_t out_n = output_dims->n;
    const int32_t out_h = output_dims->h;
    const int32_t out_w = output_dims->w;

    const int32_t in1_n = input1_dims->n;
    const int32_t in1_h = input1_dims->h;
    const int32_t in1_w = input1_dims->w;
    const int32_t in1_c = input1_dims->c;

    const int32_t in2_n = input2_dims->n;
    const int32_t in2_h = input2_dims->h;
    const int32_t in2_w = input2_dims->w;
    const int32_t in2_c = input2_dims->c;

    int32_t flat1_total = in1_n * in1_h * in1_w * in1_c;
    int32_t flat2_total = in2_n * in2_h * in2_w * in2_c;

    if (!arm_check_broadcast_required(input1_dims, input2_dims))
    {
        return arm_elementwise_squared_difference_s8(input1_data,
                                                     input2_data,
                                                     input1_offset,
                                                     input1_mult,
                                                     input1_shift,
                                                     input2_offset,
                                                     input2_mult,
                                                     input2_shift,
                                                     left_shift,
                                                     output_data,
                                                     out_offset,
                                                     out_mult,
                                                     out_shift,
                                                     out_activation_min,
                                                     out_activation_max,
                                                     flat1_total);
    }

    if (flat1_total == 1)
    {
        return arm_squared_difference_scalar_s8(input1_data,
                                                input2_data,
                                                input1_offset,
                                                input1_mult,
                                                input1_shift,
                                                input2_offset,
                                                input2_mult,
                                                input2_shift,
                                                left_shift,
                                                output_data,
                                                out_offset,
                                                out_mult,
                                                out_shift,
                                                out_activation_min,
                                                out_activation_max,
                                                flat2_total);
    }

    if (flat2_total == 1)
    {
        return arm_squared_difference_scalar_s8(input2_data,
                                                input1_data,
                                                input2_offset,
                                                input2_mult,
                                                input2_shift,
                                                input1_offset,
                                                input1_mult,
                                                input1_shift,
                                                left_shift,
                                                output_data,
                                                out_offset,
                                                out_mult,
                                                out_shift,
                                                out_activation_min,
                                                out_activation_max,
                                                flat1_total);
    }

    const int32_t wd1 = (in1_w >= in2_w) ? 0 : in1_c;
    const int32_t wd2 = (in2_w >= in1_w) ? 0 : in2_c;
    const int32_t hd1 = (in1_h >= in2_h) ? wd1 : -in1_w * (in1_c - wd1);
    const int32_t hd2 = (in2_h >= in1_h) ? wd2 : -in2_w * (in2_c - wd2);
    const int32_t bd1 = (in1_n >= in2_n) ? in1_h * in1_w * in1_c : 0;
    const int32_t bd2 = (in2_n >= in1_n) ? in2_h * in2_w * in2_c : 0;

    for (int b = 0; b < out_n; b++)
    {
        const int8_t *p1 = input1_data;
        const int8_t *p2 = input2_data;

        flat1_total = in1_h * in1_w * in1_c;
        flat2_total = in2_h * in2_w * in2_c;

        if (in1_h == in2_h && in1_w == in2_w && in1_c == in2_c)
        {
            arm_elementwise_squared_difference_s8(p1,
                                                  p2,
                                                  input1_offset,
                                                  input1_mult,
                                                  input1_shift,
                                                  input2_offset,
                                                  input2_mult,
                                                  input2_shift,
                                                  left_shift,
                                                  output_data,
                                                  out_offset,
                                                  out_mult,
                                                  out_shift,
                                                  out_activation_min,
                                                  out_activation_max,
                                                  flat1_total);
            output_data += flat1_total;
        }
        else
        {
            flat1_total = in1_w * in1_c;
            flat2_total = in2_w * in2_c;

            for (int h = 0; h < out_h; h++)
            {
                if (in1_w == in2_w && in1_c == in2_c)
                {
                    arm_elementwise_squared_difference_s8(p1,
                                                          p2,
                                                          input1_offset,
                                                          input1_mult,
                                                          input1_shift,
                                                          input2_offset,
                                                          input2_mult,
                                                          input2_shift,
                                                          left_shift,
                                                          output_data,
                                                          out_offset,
                                                          out_mult,
                                                          out_shift,
                                                          out_activation_min,
                                                          out_activation_max,
                                                          flat1_total);
                    p1 += flat1_total;
                    p2 += flat1_total;
                    output_data += flat1_total;
                }
                else if (flat1_total == 1)
                {
                    arm_squared_difference_scalar_s8(p1,
                                                     p2,
                                                     input1_offset,
                                                     input1_mult,
                                                     input1_shift,
                                                     input2_offset,
                                                     input2_mult,
                                                     input2_shift,
                                                     left_shift,
                                                     output_data,
                                                     out_offset,
                                                     out_mult,
                                                     out_shift,
                                                     out_activation_min,
                                                     out_activation_max,
                                                     flat2_total);
                    p2 += flat2_total;
                    output_data += flat2_total;
                }
                else if (flat2_total == 1)
                {
                    arm_squared_difference_scalar_s8(p2,
                                                     p1,
                                                     input2_offset,
                                                     input2_mult,
                                                     input2_shift,
                                                     input1_offset,
                                                     input1_mult,
                                                     input1_shift,
                                                     left_shift,
                                                     output_data,
                                                     out_offset,
                                                     out_mult,
                                                     out_shift,
                                                     out_activation_min,
                                                     out_activation_max,
                                                     flat1_total);
                    p1 += flat1_total;
                    output_data += flat1_total;
                }
                else
                {
                    for (int w = 0; w < out_w; w++)
                    {
                        if (in1_c == in2_c)
                        {
                            arm_elementwise_squared_difference_s8(p1,
                                                                  p2,
                                                                  input1_offset,
                                                                  input1_mult,
                                                                  input1_shift,
                                                                  input2_offset,
                                                                  input2_mult,
                                                                  input2_shift,
                                                                  left_shift,
                                                                  output_data,
                                                                  out_offset,
                                                                  out_mult,
                                                                  out_shift,
                                                                  out_activation_min,
                                                                  out_activation_max,
                                                                  in1_c);
                            p1 += in1_c;
                            p2 += in1_c;
                            output_data += in1_c;
                        }
                        else if (in1_c == 1)
                        {
                            arm_squared_difference_scalar_s8(p1,
                                                             p2,
                                                             input1_offset,
                                                             input1_mult,
                                                             input1_shift,
                                                             input2_offset,
                                                             input2_mult,
                                                             input2_shift,
                                                             left_shift,
                                                             output_data,
                                                             out_offset,
                                                             out_mult,
                                                             out_shift,
                                                             out_activation_min,
                                                             out_activation_max,
                                                             in2_c);
                            p1++;
                            p2 += in2_c;
                            output_data += in2_c;
                        }
                        else
                        {
                            arm_squared_difference_scalar_s8(p2,
                                                             p1,
                                                             input2_offset,
                                                             input2_mult,
                                                             input2_shift,
                                                             input1_offset,
                                                             input1_mult,
                                                             input1_shift,
                                                             left_shift,
                                                             output_data,
                                                             out_offset,
                                                             out_mult,
                                                             out_shift,
                                                             out_activation_min,
                                                             out_activation_max,
                                                             in1_c);
                            p1 += in1_c;
                            p2++;
                            output_data += in1_c;
                        }

                        p1 -= wd1;
                        p2 -= wd2;
                    }
                }

                p1 += hd1;
                p2 += hd2;
            }
        }

        input1_data += bd1;
        input2_data += bd2;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */