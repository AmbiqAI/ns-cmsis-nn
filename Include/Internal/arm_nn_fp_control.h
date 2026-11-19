/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#ifndef ARM_NN_FP_CONTROL_H
#define ARM_NN_FP_CONTROL_H

#include <stdint.h>

/* Floating-point capability alone does not distinguish FPSCR from FPCR. */
#if defined(__ARM_FP) && (__ARM_FP != 0) && (defined(__aarch64__) || defined(__arm__))
    #define ARM_NN_FP_CONTROL_AVAILABLE 1

    #if defined(__aarch64__)
typedef uint64_t arm_nn_fp_control_t;
    #else
typedef uint32_t arm_nn_fp_control_t;
    #endif

static inline arm_nn_fp_control_t arm_nn_get_fp_control(void)
{
    arm_nn_fp_control_t value;
    #if defined(__aarch64__)
    __asm__ volatile("mrs %0, fpcr" : "=r"(value));
    #else
    __asm__ volatile("vmrs %0, fpscr" : "=r"(value));
    #endif
    return value;
}

static inline void arm_nn_set_fp_control(arm_nn_fp_control_t value)
{
    #if defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0" : : "r"(value) : "memory");
    #else
    __asm__ volatile("vmsr fpscr, %0" : : "r"(value) : "memory");
    #endif
}
#endif
#endif
