// SPDX-FileCopyrightText: Copyright 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#ifndef ARM_NN_VECTOR_SUM_S8_MVE_H
#define ARM_NN_VECTOR_SUM_S8_MVE_H

#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_MVEI)
typedef struct
{
    int32_t sum[5];
} arm_nn_vector_sum_s8x5_result;

/* Integer address operands keep final post-increments out of C pointer arithmetic. */
__STATIC_FORCEINLINE arm_nn_vector_sum_s8x5_result arm_nn_vector_sum_s8x5_mve(const int8_t *data, int32_t cols)
{
    arm_nn_vector_sum_s8x5_result result = {{0, 0, 0, 0, 0}};
    if (cols > 0)
    {
        register int32_t s0 = 0;
        register int32_t s1 = 0;
        register int32_t s2 = 0;
        register int32_t s3 = 0;
        register int32_t s4 = 0;
        register uintptr_t p0 = (uintptr_t)data;
        register uintptr_t p1 = (uintptr_t)(data + cols);
        register uintptr_t p2 = (uintptr_t)(data + 2 * cols);
        register uintptr_t p3 = (uintptr_t)(data + 3 * cols);
        register uintptr_t p4 = (uintptr_t)(data + 4 * cols);
        register uint32_t saved_predicate;
        /* cols is consumed before any read/write operand changes. */
        __asm volatile("vmrs %[saved], p0\n"
                       "dlstp.8 lr, %[cols]\n"
                       "1: vldrb.u8 q0, [%[p0]], #16\n"
                       "vaddva.s8 %[s0], q0\n"
                       "vldrb.u8 q0, [%[p1]], #16\n"
                       "vaddva.s8 %[s1], q0\n"
                       "vldrb.u8 q0, [%[p2]], #16\n"
                       "vaddva.s8 %[s2], q0\n"
                       "vldrb.u8 q0, [%[p3]], #16\n"
                       "vaddva.s8 %[s3], q0\n"
                       "vldrb.u8 q0, [%[p4]], #16\n"
                       "vaddva.s8 %[s4], q0\n"
                       "letp lr, 1b\n"
                       "vmsr p0, %[saved]\n"
                       : [saved] "=&r"(saved_predicate),
                         [p0] "+r"(p0),
                         [p1] "+r"(p1),
                         [p2] "+r"(p2),
                         [p3] "+r"(p3),
                         [p4] "+r"(p4),
                         [s0] "+Te"(s0),
                         [s1] "+Te"(s1),
                         [s2] "+Te"(s2),
                         [s3] "+Te"(s3),
                         [s4] "+Te"(s4)
                       : [cols] "r"(cols)
                       : "q0", "lr", "cc", "memory");
        result.sum[0] = s0;
        result.sum[1] = s1;
        result.sum[2] = s2;
        result.sum[3] = s3;
        result.sum[4] = s4;
    }
    return result;
}

__STATIC_FORCEINLINE int32_t arm_nn_vector_sum_s8x1_mve(const int8_t *data, int32_t cols)
{
    register int32_t sum = 0;
    if (cols > 0)
    {
        register uintptr_t address = (uintptr_t)data;
        register uint32_t saved_predicate;
        /* cols is consumed before any read/write operand changes. */
        __asm volatile("vmrs %[saved], p0\n"
                       "dlstp.8 lr, %[cols]\n"
                       "1: vldrb.u8 q0, [%[address]], #16\n"
                       "vaddva.s8 %[sum], q0\n"
                       "letp lr, 1b\n"
                       "vmsr p0, %[saved]\n"
                       : [saved] "=&r"(saved_predicate), [address] "+r"(address), [sum] "+Te"(sum)
                       : [cols] "r"(cols)
                       : "q0", "lr", "cc", "memory");
    }
    return sum;
}
#endif
#endif
