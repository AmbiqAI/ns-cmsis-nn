/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/**
 * @file arm_nn_build_target.h
 * @brief The build of ns-cmsis-nn the including translation unit compiles for (RFC #829).
 *
 * Derived from the compiler's own macros through the library's headers, so it describes exactly the build the kernels
 * see. Every ARM_NN_BUILD_* value is a literal 0/1 (or an ARM_NN_ISA_* / ARM_NN_TOOLCHAIN_ID_* value) usable in \#if.
 * ARM_NN_BUILD_HAS_* names one feature token of Tests/KernelContracts/kernel_implementations.json each.
 * arm_nn_build_target_check.h compares a plan with this build; the library defines ARM_NN_BUILD_STAMP for its own.
 */

#ifndef ARM_NN_BUILD_TARGET_H
#define ARM_NN_BUILD_TARGET_H

#include "Internal/arm_nn_compiler.h"
#include "arm_nn_math_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** ISA without ARM_MATH_DSP or ARM_MATH_MVEI (Cortex-M0/M0+); arm_nn_plan_isa value. */
#define ARM_NN_ISA_SCALAR 0
/** ISA with ARM_MATH_DSP and without ARM_MATH_MVEI (Cortex-M4/M7/M33); arm_nn_plan_isa value. */
#define ARM_NN_ISA_DSP 1
/** ISA with ARM_MATH_MVEI (Cortex-M55/M85); arm_nn_plan_isa value. */
#define ARM_NN_ISA_MVE 2

/** Any compiler that is not Arm Compiler 6; arm_nn_plan_toolchain value. */
#define ARM_NN_TOOLCHAIN_ID_GCC 0
/** Arm Compiler 6; arm_nn_plan_toolchain value. */
#define ARM_NN_TOOLCHAIN_ID_ARMCLANG 1

#if defined(ARM_MATH_MVEI)
    /** Manifest feature mvei: ARM_MATH_MVEI, also under ARM_MATH_AUTOVECTORIZE. */
    #define ARM_NN_BUILD_HAS_MVEI 1
#else
    #define ARM_NN_BUILD_HAS_MVEI 0
#endif

#if defined(ARM_MATH_DSP)
    /** Manifest feature dsp: ARM_MATH_DSP, which MVE builds define too. */
    #define ARM_NN_BUILD_HAS_DSP 1
#else
    #define ARM_NN_BUILD_HAS_DSP 0
#endif

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    /** Manifest feature mvef: the float MVE paths, ARM_MATH_MVEF without ARM_MATH_AUTOVECTORIZE. */
    #define ARM_NN_BUILD_HAS_MVEF 1
#else
    #define ARM_NN_BUILD_HAS_MVEF 0
#endif

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /** Manifest feature mve_f16: ARM_MATH_MVE_FLOAT16 without ARM_MATH_AUTOVECTORIZE. */
    #define ARM_NN_BUILD_HAS_MVE_F16 1
#else
    #define ARM_NN_BUILD_HAS_MVE_F16 0
#endif

#if ARM_NN_ENABLE_F32
    /** Manifest feature f32_api: ARM_NN_ENABLE_F32. */
    #define ARM_NN_BUILD_HAS_F32_API 1
#else
    #define ARM_NN_BUILD_HAS_F32_API 0
#endif

#if ARM_NN_ENABLE_F16
    /** Manifest feature f16_api: ARM_NN_ENABLE_F16. */
    #define ARM_NN_BUILD_HAS_F16_API 1
#else
    #define ARM_NN_BUILD_HAS_F16_API 0
#endif

#if defined(ARM_MATH_AUTOVECTORIZE)
    /** Manifest feature autovectorize: ARM_MATH_AUTOVECTORIZE. */
    #define ARM_NN_BUILD_HAS_AUTOVECTORIZE 1
#else
    #define ARM_NN_BUILD_HAS_AUTOVECTORIZE 0
#endif

#if defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    /** Manifest feature single_rounding: CMSIS_NN_USE_SINGLE_ROUNDING. */
    #define ARM_NN_BUILD_HAS_SINGLE_ROUNDING 1
#else
    #define ARM_NN_BUILD_HAS_SINGLE_ROUNDING 0
#endif

#if ARM_NN_BUILD_HAS_MVEI
    /** The ISA of this build, an ARM_NN_ISA_* value. */
    #define ARM_NN_BUILD_ISA 2
#elif ARM_NN_BUILD_HAS_DSP
    #define ARM_NN_BUILD_ISA 1
#else
    #define ARM_NN_BUILD_ISA 0
#endif

#if ARM_NN_TOOLCHAIN_ARMCLANG
    /** The toolchain this build routes and sizes for, an ARM_NN_TOOLCHAIN_ID_* value. */
    #define ARM_NN_BUILD_TOOLCHAIN 1
#else
    #define ARM_NN_BUILD_TOOLCHAIN 0
#endif

#define ARM_NN_BUILD_ISA_NAME_0 scalar
#define ARM_NN_BUILD_ISA_NAME_1 dsp
#define ARM_NN_BUILD_ISA_NAME_2 mve
#define ARM_NN_BUILD_TOOLCHAIN_NAME_0 gcc
#define ARM_NN_BUILD_TOOLCHAIN_NAME_1 armclang
#define ARM_NN_BUILD_STAMP_PASTE(isa, av, sr, tc) arm_nn_build_##isa##_av##av##_sr##sr##_##tc
#define ARM_NN_BUILD_STAMP_NAMED(isa, av, sr, tc) ARM_NN_BUILD_STAMP_PASTE(isa, av, sr, tc)
#define ARM_NN_BUILD_STAMP_ISA(isa) ARM_NN_BUILD_ISA_NAME_##isa
#define ARM_NN_BUILD_STAMP_TOOLCHAIN(tc) ARM_NN_BUILD_TOOLCHAIN_NAME_##tc
#define ARM_NN_BUILD_STAMP_EXPANDED(isa, av, sr, tc)                                                                   \
    ARM_NN_BUILD_STAMP_NAMED(ARM_NN_BUILD_STAMP_ISA(isa), av, sr, ARM_NN_BUILD_STAMP_TOOLCHAIN(tc))

/**
 * @brief Symbol name of the library stamp of a build, such as arm_nn_build_mve_av0_sr0_gcc.
 *
 * isa is an ARM_NN_ISA_* value, av and sr are 0 or 1 (autovectorize, single rounding) and tc is an
 * ARM_NN_TOOLCHAIN_ID_* value; each must expand to a bare literal.
 */
#define ARM_NN_BUILD_STAMP_OF(isa, av, sr, tc) ARM_NN_BUILD_STAMP_EXPANDED(isa, av, sr, tc)

/** Symbol name of this build's stamp, which the library defines in arm_nn_build_target.c. */
#define ARM_NN_BUILD_STAMP                                                                                             \
    ARM_NN_BUILD_STAMP_OF(                                                                                             \
        ARM_NN_BUILD_ISA, ARM_NN_BUILD_HAS_AUTOVECTORIZE, ARM_NN_BUILD_HAS_SINGLE_ROUNDING, ARM_NN_BUILD_TOOLCHAIN)

/** Stamp of the build the library was compiled for; always 1. A library of another build defines another name. */
extern const uint8_t ARM_NN_BUILD_STAMP;

#ifdef __cplusplus
}
#endif

#endif /* ARM_NN_BUILD_TARGET_H */
