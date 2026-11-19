/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#ifndef ARM_NN_DIV_FLT_H
#define ARM_NN_DIV_FLT_H

#include "arm_nn_math_types.h"

/* Clang's Arm target cannot disable all fast-math assumptions with a source
 * pragma. CMake sets this per-file flag; pack/manual consumers must do so too. */
#if defined(__clang__)
    #if (ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16) && (defined(__FAST_MATH__) || (__FINITE_MATH_ONLY__ > 0))
        #error "Compile the DIV sources with -fno-fast-math (including pack/manual builds)"
    #endif
#elif defined(__GNUC__)
    #pragma GCC optimize("no-fast-math")
#endif

#include "Internal/arm_nn_activation_flt.h"
#include "Internal/arm_nn_broadcast_walk.h"
#include <limits.h>
#include <stdbool.h>

static inline bool
arm_nn_div_dims_valid(const cmsis_nn_dims *a, const cmsis_nn_dims *b, const cmsis_nn_dims *out, int32_t element_size)
{
    if (!a || !b || !out || !arm_nn_broadcast_dims_valid(a, b, out))
    {
        return false;
    }
    /* Each input extent is at most the output extent, so bounding the output
     * also bounds every input stride and product used by the broadcast walk. */
    int32_t count = 1;
    const int32_t dims[4] = {out->n, out->h, out->w, out->c};
    for (int i = 0; i < 4; ++i)
    {
        if (count > (INT32_MAX / element_size) / dims[i])
        {
            return false;
        }
        count *= dims[i];
    }
    return true;
}

#endif /* ARM_NN_DIV_FLT_H */
