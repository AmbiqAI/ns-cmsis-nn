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
 * Title:        arm_nn_rbf_kernel_f32
 * Description:  The radial basis function kernel of a float32 reference table against one row, in one stage
 *
 * $Date:        15 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_rbf_kernel_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32
    #include <stdint.h>

/* Compare byte distances, never add lengths to addresses that could wrap. */
static bool rbf_kernel_overlaps(const void *a, size_t a_size, const void *b, size_t b_size)
{
    const uintptr_t left = (uintptr_t)a, right = (uintptr_t)b;
    return left < right ? right - left < a_size : left - right < b_size;
}

/** @ingroup Reduction */
arm_cmsis_nn_status arm_nn_rbf_kernel_f32(const float32_t *input,
                                          const float32_t *references,
                                          int32_t rows,
                                          int32_t features,
                                          float32_t neg_gamma,
                                          float32_t *output)
{
    if (!input || !references || !output || rows <= 0 || features <= 0 ||
        rows > (INT32_MAX / (int32_t)sizeof(float32_t)) / features)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const size_t x_bytes = (size_t)features * sizeof(float32_t);
    const size_t out_bytes = (size_t)rows * sizeof(float32_t);
    if (rbf_kernel_overlaps(input, x_bytes, output, out_bytes) ||
        rbf_kernel_overlaps(references, (size_t)rows * x_bytes, output, out_bytes))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_nn_rbf_kernel_body_f32(input, references, rows, features, neg_gamma, output);
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F32 */
