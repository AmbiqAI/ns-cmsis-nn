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
 * Title:        arm_sqrt_s8
 * Description:  Elementwise square root
 *
 * $Date:        6 March 2026
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
 * s8 elementwise square root
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_sqrt_s8(const int8_t *input,
                                const cmsis_nn_dims *input_dims,
                                int8_t *output,
                                int8_t *sqrt_lut)
{
    const int32_t block_size = input_dims->n * input_dims->h * input_dims->w * input_dims->c;

#if defined(ARM_MATH_MVEI)
    int32_t loop_cnt = block_size;

    while (loop_cnt > 0)
    {
        const mve_pred16_t p = vctp8q((uint32_t)loop_cnt);
        const uint8x16_t lut_indices = (uint8x16_t)vldrbq_z_s8(input, p);
        const int8x16_t out = vldrbq_gather_offset_z_s8(sqrt_lut, lut_indices, p);
        vstrbq_p_s8(output, out, p);

        input += 16;
        output += 16;
        loop_cnt -= 16;
    }
#else
    int32_t i = 0;
    for (; i <= (block_size - 4); i += 4)
    {
        output[i] = sqrt_lut[(uint8_t)input[i]];
        output[i + 1] = sqrt_lut[(uint8_t)input[i + 1]];
        output[i + 2] = sqrt_lut[(uint8_t)input[i + 2]];
        output[i + 3] = sqrt_lut[(uint8_t)input[i + 3]];
    }

    for (; i < block_size; ++i)
    {
        output[i] = sqrt_lut[(uint8_t)input[i]];
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
