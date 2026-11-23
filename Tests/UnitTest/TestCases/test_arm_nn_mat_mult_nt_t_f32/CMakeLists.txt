#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

add_cmsis_nn_unit_test_executable(test_arm_nn_mat_mult_nt_t_f32)

target_sources(test_arm_nn_mat_mult_nt_t_f32 PRIVATE
    Unity/unity_test_arm_nn_mat_mult_nt_t_f32.c
    Unity/TestRunner/unity_test_arm_nn_mat_mult_nt_t_f32_runner.c
    mat_mult_nt_t_f32_previous.c
    mat_mult_nt_t_f32_literal_library.c
    mat_mult_nt_t_f32_literal_c11.c)

# The previous body and the library-list literal functions take the library's own options, which follow the
# suite's: -fno-finite-math-only, which the suite needs for Unity's NaN checks, is undone where the library's
# -Ofast implies finite math (GCC; the Clang configuration replaces -Ofast with -O3). Each later option wins.
set(_mm_library_options ${CMSIS_OPTIMIZATION_LEVEL})
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMSIS_OPTIMIZATION_LEVEL MATCHES "Ofast")
    list(APPEND _mm_library_options -ffinite-math-only)
endif()
set_source_files_properties(mat_mult_nt_t_f32_previous.c PROPERTIES COMPILE_OPTIONS "${_mm_library_options}")
# Arm GNU 14.2.1 at -Ofast, on the scalar leg, warns -Waggressive-loop-optimizations ("iteration N invokes undefined
# behavior") in the remainder row loop of the literal shapes 32 x 16 and 2048 x 16, where that loop never runs. The
# warning stays visible but is not an error here, so the cases run.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(_mm_literal_options ${_mm_library_options} -Wno-error=aggressive-loop-optimizations)
else()
    set(_mm_literal_options ${_mm_library_options})
endif()
set_source_files_properties(mat_mult_nt_t_f32_literal_library.c PROPERTIES COMPILE_OPTIONS "${_mm_literal_options}")
# A generated module's list in the parity recipe: ISO C11 at -O2, no fast-math, so GCC does not contract.
set_source_files_properties(mat_mult_nt_t_f32_literal_c11.c PROPERTIES COMPILE_OPTIONS "-std=c11;-O2")
