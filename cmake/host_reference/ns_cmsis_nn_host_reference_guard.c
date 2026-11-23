/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

/*
 * Compiled into the host reference library with the library's own flags and
 * definitions, so a kernel path selected on the target itself (which the
 * configure probes in cmake/ns_cmsis_nn_host_reference.cmake cannot see) still
 * fails the build. See cmake/ns_cmsis_nn_host_reference.cmake.
 */

#include "arm_nn_math_types.h"

#if defined(ARM_MATH_DSP) || defined(ARM_MATH_MVEI) || defined(ARM_MATH_MVEF) || defined(ARM_MATH_HELIUM) ||           \
    defined(ARM_MATH_MVE_FLOAT16) || defined(ARM_MATH_AUTOVECTORIZE) || defined(ARM_MATH_NEON)
    #error "ns-cmsis-nn host reference: a kernel path macro is defined; the reference is the plain C build"
#endif

#if !(ARM_NN_ENABLE_F32 && ARM_NN_ENABLE_F16)
    #error "ns-cmsis-nn host reference: ARM_NN_ENABLE_F32 and ARM_NN_ENABLE_F16 must both be 1"
#endif

#if defined(__FAST_MATH__)
    #error "ns-cmsis-nn host reference: built with fast-math"
#endif

const char ns_cmsis_nn_host_reference_marker[] = "ns-cmsis-nn host reference";
