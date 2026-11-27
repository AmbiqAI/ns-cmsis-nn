/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/**
 * @file arm_nn_build_target_check.h
 * @brief Fails the build when code planned for one ns-cmsis-nn build is compiled or linked for another (RFC #829).
 *
 * Define the plan's build, then include this header:
 *
 *     #define ARM_NN_EXPECT_ISA ARM_NN_ISA_MVE         (an ARM_NN_ISA_* value)
 *     #define ARM_NN_EXPECT_AUTOVECTORIZE 0            (1 when planned with ARM_MATH_AUTOVECTORIZE)
 *     #define ARM_NN_EXPECT_SINGLE_ROUNDING 0          (1 when planned with CMSIS_NN_USE_SINGLE_ROUNDING)
 *     #define ARM_NN_EXPECT_TOOLCHAIN ARM_NN_TOOLCHAIN_ID_GCC
 *     #include "arm_nn_build_target_check.h"
 *
 * A compiler that builds anything else stops with an \#error naming the difference. ARM_NN_LIBRARY_STAMP() reads the
 * stamp a library of the planned build defines, so linking a library built for another target fails with an undefined
 * reference to the planned stamp (such as arm_nn_build_mve_av0_sr0_gcc); call it once from code that is
 * linked, such as the model's init. ns_cmsis_nn_contracts.render_guard() writes this block for a plan.
 */

#ifndef ARM_NN_BUILD_TARGET_CHECK_H
#define ARM_NN_BUILD_TARGET_CHECK_H

#include "arm_nn_build_target.h"

#if !defined(ARM_NN_EXPECT_ISA) || !defined(ARM_NN_EXPECT_AUTOVECTORIZE) || !defined(ARM_NN_EXPECT_SINGLE_ROUNDING) || \
    !defined(ARM_NN_EXPECT_TOOLCHAIN)
    #error "ns-cmsis-nn build check: define ARM_NN_EXPECT_ISA, _AUTOVECTORIZE, _SINGLE_ROUNDING and _TOOLCHAIN first"
#elif (ARM_NN_EXPECT_ISA != 0 && ARM_NN_EXPECT_ISA != 1 && ARM_NN_EXPECT_ISA != 2) ||                                  \
    (ARM_NN_EXPECT_AUTOVECTORIZE != 0 && ARM_NN_EXPECT_AUTOVECTORIZE != 1) ||                                          \
    (ARM_NN_EXPECT_SINGLE_ROUNDING != 0 && ARM_NN_EXPECT_SINGLE_ROUNDING != 1) ||                                      \
    (ARM_NN_EXPECT_TOOLCHAIN != 0 && ARM_NN_EXPECT_TOOLCHAIN != 1)
    #error "ns-cmsis-nn build check: an ARM_NN_EXPECT_* value is out of range (ISA 0-2, the others 0 or 1)"
#elif ARM_NN_EXPECT_AUTOVECTORIZE && ARM_NN_EXPECT_ISA != ARM_NN_ISA_MVE
    #error "ns-cmsis-nn build check: ARM_NN_EXPECT_AUTOVECTORIZE applies to an MVE plan only"
#else

    #if ARM_NN_EXPECT_ISA != ARM_NN_BUILD_ISA
        #if ARM_NN_EXPECT_ISA == ARM_NN_ISA_MVE && ARM_NN_BUILD_ISA == ARM_NN_ISA_DSP
            #error "ns-cmsis-nn build check: planned for an mve library, this compiler builds dsp (no ARM_MATH_MVEI)"
        #elif ARM_NN_EXPECT_ISA == ARM_NN_ISA_MVE
            #error "ns-cmsis-nn build check: planned for an mve library, this compiler builds scalar (no ARM_MATH_MVEI)"
        #elif ARM_NN_EXPECT_ISA == ARM_NN_ISA_DSP && ARM_NN_BUILD_ISA == ARM_NN_ISA_MVE
            #error "ns-cmsis-nn build check: planned for a dsp library, this compiler builds mve (ARM_MATH_MVEI)"
        #elif ARM_NN_EXPECT_ISA == ARM_NN_ISA_DSP
            #error "ns-cmsis-nn build check: planned for a dsp library, this compiler builds scalar (no ARM_MATH_DSP)"
        #elif ARM_NN_BUILD_ISA == ARM_NN_ISA_MVE
            #error "ns-cmsis-nn build check: planned for a scalar library, this compiler builds mve (ARM_MATH_MVEI)"
        #else
            #error "ns-cmsis-nn build check: planned for a scalar library, this compiler builds dsp (ARM_MATH_DSP)"
        #endif
    #endif

    #if ARM_NN_EXPECT_AUTOVECTORIZE && !ARM_NN_BUILD_HAS_AUTOVECTORIZE
        #error "ns-cmsis-nn build check: planned with ARM_MATH_AUTOVECTORIZE, this build does not define it"
    #elif !ARM_NN_EXPECT_AUTOVECTORIZE && ARM_NN_BUILD_HAS_AUTOVECTORIZE
        #error "ns-cmsis-nn build check: planned without ARM_MATH_AUTOVECTORIZE, this build defines it"
    #endif

    #if ARM_NN_EXPECT_SINGLE_ROUNDING && !ARM_NN_BUILD_HAS_SINGLE_ROUNDING
        #error "ns-cmsis-nn build check: planned with CMSIS_NN_USE_SINGLE_ROUNDING, this build does not define it"
    #elif !ARM_NN_EXPECT_SINGLE_ROUNDING && ARM_NN_BUILD_HAS_SINGLE_ROUNDING
        #error "ns-cmsis-nn build check: planned without CMSIS_NN_USE_SINGLE_ROUNDING, this build defines it"
    #endif

    #if ARM_NN_EXPECT_TOOLCHAIN != ARM_NN_BUILD_TOOLCHAIN
        #if ARM_NN_EXPECT_TOOLCHAIN == ARM_NN_TOOLCHAIN_ID_ARMCLANG
            #error "ns-cmsis-nn build check: planned for Arm Compiler 6 (armclang), this compiler is not armclang"
        #else
            #error "ns-cmsis-nn build check: planned for a gcc library, this compiler is Arm Compiler 6 (armclang)"
        #endif
    #endif

    /** Symbol name of the stamp a library of the planned build defines. */
    #define ARM_NN_EXPECT_STAMP                                                                                        \
        ARM_NN_BUILD_STAMP_OF(                                                                                         \
            ARM_NN_EXPECT_ISA, ARM_NN_EXPECT_AUTOVECTORIZE, ARM_NN_EXPECT_SINGLE_ROUNDING, ARM_NN_EXPECT_TOOLCHAIN)

    #ifdef __cplusplus
extern "C" {
    #endif
/** Stamp of the planned build, defined only by a library compiled for it. */
extern const uint8_t ARM_NN_EXPECT_STAMP;
    #ifdef __cplusplus
}
    #endif

    /** Reads the planned build's library stamp (1); the reference fails to link against a library of another build. */
    #define ARM_NN_LIBRARY_STAMP() (*(const volatile uint8_t *)&ARM_NN_EXPECT_STAMP)

#endif

#endif /* ARM_NN_BUILD_TARGET_CHECK_H */
