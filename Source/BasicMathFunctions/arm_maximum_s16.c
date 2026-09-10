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
 * Title:        arm_maximum_s16
 * Description:  Minimum and Maximum
 *
 * $Date:        17 February 2025
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
 * @addtogroup minimumMaximum
 * @{
 */

static arm_cmsis_nn_status
arm_max_no_broadcast_s16(const int16_t *input_1, const int16_t *input_2, int16_t *output, int32_t flat_size)
{
#if defined(ARM_MATH_MVEI)

    __ASM volatile(" .p2align 2                              \n"
                   "   wlstp.16         lr, %[cnt], 1f       \n"
                   "2:                                       \n"
                   "   vldrh.16         q0, [%[in1]], #16    \n"
                   "   vldrh.16         q1, [%[in2]], #16    \n"
                   "   vmax.s16         q2, q0, q1           \n"
                   "   vstrh.16         q2, [%[out]], #16    \n"
                   "   letp            lr, 2b                \n"
                   "1:                                       \n"
                   : [in1] "+r"(input_1), [in2] "+r"(input_2), [out] "+r"(output)
                   : [cnt] "r"(flat_size)
                   : "q0", "q1", "q2", "memory", "r14");

#else
    while (flat_size > 0)
    {
        int16_t in1 = *input_1++;
        int16_t in2 = *input_2++;
        *output++ = in1 >= in2 ? in1 : in2;
        --flat_size;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status
arm_max_scalar_s16(const int16_t *input_1, const int16_t *input_2, int16_t *output, int32_t flat_size)
{
#if defined(ARM_MATH_MVEI)

    __ASM volatile(" .p2align 2                              \n"
                   "   wlstp.16         lr, %[cnt], 1f       \n"
                   "   vdup.16          q0, %[in1]           \n"
                   "2:                                       \n"
                   "   vldrh.16         q1, [%[in2]], #16    \n"
                   "   vmax.s16         q2, q0, q1           \n"
                   "   vstrh.16         q2, [%[out]], #16    \n"
                   "   letp             lr, 2b               \n"
                   "1:                                       \n"
                   : [in2] "+r"(input_2), [out] "+r"(output)
                   : [in1] "r"(*input_1), [cnt] "r"(flat_size)
                   : "q0", "q1", "q2", "memory", "r14");

#else
    int16_t in1 = *input_1;
    while (flat_size > 0)
    {
        int16_t in2 = *input_2++;
        *output++ = in1 >= in2 ? in1 : in2;
        --flat_size;
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * s16 maximum
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_maximum_s16(const cmsis_nn_context *ctx,
                                    const int16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const int16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    int16_t *output_data,
                                    const cmsis_nn_dims *output_dims)
{
    (void)ctx;
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ARM_NN_BROADCAST_WALK_NHWC(int16_t,
                               int16_t,
                               input_1_data,
                               input_1_dims,
                               input_2_data,
                               input_2_dims,
                               output_data,
                               output_dims,
                               arm_max_no_broadcast_s16,
                               arm_max_scalar_s16,
                               arm_max_scalar_s16);

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
