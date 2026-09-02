#
# SPDX-FileCopyrightText: Copyright 2019-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.15.6)

project(cmsis_nn_unit_tests VERSION 0.0.1)

set(CMSIS_PATH "</path/to/CMSIS>" CACHE PATH "Path to CMSIS.")

add_compile_options(-Werror
                    -Wimplicit-function-declaration
                    -Wunused-variable
                    -Wunused-function
                    -Wno-redundant-decls
                    -Wvla)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wjump-misses-init)
endif()

option(BUILD_CMSIS_NN_UNIT "If building the unit tests from another project, i.e. \
platform dependencies need to be provided externally." OFF)

if (${CMSIS_PATH} STREQUAL "</path/to/CMSIS>")
  message(FATAL_ERROR "CMSIS_PATH not set. Did you provide -DCMSIS_PATH=<path/to/CMSIS>?")
endif()

if(NOT BUILD_CMSIS_NN_UNIT)
    set(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300 ON)
    # Only target builds omit the frame pointer. Applying it to host builds too would
    # override a -fno-omit-frame-pointer passed in CMAKE_C_FLAGS, because add_compile_options
    # lands after CMAKE_C_FLAGS on the command line and the last of the pair wins.
    add_compile_options(-fomit-frame-pointer)
else()
    set(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300 OFF)
endif()

if(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
    set(FVP_CORSTONE_300_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300" CACHE PATH
        "Dependencies for using FVP based on Arm Corstone-300 software.")
    set(CMAKE_EXECUTABLE_SUFFIX ".elf")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    # The host Clang regression build treats -Ofast as deprecated and Unity
    # relies on isnan/isinf checks, so use a conforming optimization level.
    set(CMSIS_OPTIMIZATION_LEVEL "-O3" CACHE STRING "Compiler optimization level." FORCE)
endif()

# Build the functions to be tested.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cmsis-nn)
if(NOT BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(cmsis-nn PRIVATE -Wno-error=unknown-attributes)
endif()
# The library above keeps its own optimisation level. Everything configured below inherits these
# directory-level options -- the Unity harness and the suites, and on the Corstone-300 build the
# retarget/uart shims and the CMSIS startup/system objects too -- and so is built with finite-math
# assumptions disabled on every compiler. Unity's TEST_ASSERT_FLOAT_IS_NAN/IS_INF go through
# UNITY_IS_NAN/UNITY_IS_INF, which with a math.h in scope expand to __builtin_isnan() and
# __builtin_isinf_sign() (`n != n` is only Unity's fallback when math.h supplies no isnan macro).
# -ffinite-math-only, implied by -Ofast, folds both to false, and Unity's failure printer -- guarded
# by the same builtins -- then loops forever on the non-finite value it is asked to print. On a
# Cortex-M0 build that hangs test_arm_reduce_sum_f32's NaN/Inf case indefinitely; on the host it
# reproduces at -Ofast. armclang goes further and rejects the INFINITY/NAN literals in those suites
# outright at -Ofast (-Werror,-Wnan-infinity-disabled). The four non-test objects that also take the
# flag -- Corstone-300's retarget.c and uart.c, and CMSIS_5's startup_ARMCM*.c and system_ARMCM*.c --
# are unchanged in code: only their .ARM.attributes Tag_ABI_FP_number_model tag moves from Finite to
# IEEE 754.
#
# Nothing on a pull request exercises this flag. It only bites where -Ofast (or -ffast-math) is in
# force, which is the Corstone-300 branch above; the host job builds at -O1, where the flag is a
# no-op -- __FINITE_MATH_ONLY__ is already 0 with or without it, and stripping it there leaves the
# host suites byte-for-byte green. And legacy-tester.yml, which does compile at -Ofast, is
# workflow_call/workflow_dispatch only and is invoked solely by release.yml's release-unit-tests.
# So removing this flag passes every check on a PR and surfaces as a hang on the FVP during a
# release -- do not remove it on the grounds that CI stays green.
if(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
    add_compile_options(${CMSIS_OPTIMIZATION_LEVEL} -fno-finite-math-only)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    # The host Clang build also swaps the deprecated -Ofast for -O3.
    add_compile_options(-O3
                        -fno-finite-math-only
                        -Wno-error=unknown-attributes)
else()
    add_compile_options(${CMSIS_OPTIMIZATION_LEVEL} -fno-finite-math-only)
endif()

enable_testing()

# Target for all unit tests.
add_custom_target(cmsis_nn_unit_tests)

# This function should be used instead of add_executable.
set_property(GLOBAL PROPERTY cmsis_nn_unit_test_executables)
function(add_cmsis_nn_unit_test_executable)
    get_property(tmp GLOBAL PROPERTY cmsis_nn_unit_test_executables)
    foreach(target ${ARGV})
        set(tmp "${tmp} ${target}")
        add_executable(${target})
        if (BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_options(${target} PRIVATE "--specs=nosys.specs")
        endif()
        add_dependencies(cmsis_nn_unit_tests ${target})
        if(NOT BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
            # Host builds run every suite through ctest; the FVP flow runs the .elf images itself.
            add_test(NAME ${target} COMMAND ${target})
        endif()
    endforeach()
    set_property(GLOBAL PROPERTY cmsis_nn_unit_test_executables "${tmp}")
endfunction(add_cmsis_nn_unit_test_executable)

add_subdirectory(TestCases/test_arm_avgpool_s16)
add_subdirectory(TestCases/test_arm_avgpool_s8)
add_subdirectory(TestCases/test_arm_batch_matmul_s8)
add_subdirectory(TestCases/test_arm_batch_matmul_s16)
add_subdirectory(TestCases/test_arm_convolve_1x1_s8_fast)
add_subdirectory(TestCases/test_arm_convolve_1x1_s4_fast)
add_subdirectory(TestCases/test_arm_convolve_s16)
add_subdirectory(TestCases/test_arm_convolve_s8)
add_subdirectory(TestCases/test_arm_convolve_s4)
add_subdirectory(TestCases/test_arm_convolve_1_x_n_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_3x3_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_fast_s16)
add_subdirectory(TestCases/test_arm_depthwise_conv_s16)
add_subdirectory(TestCases/test_arm_depthwise_conv_s4)
add_subdirectory(TestCases/test_arm_depthwise_conv_s4_opt)
add_subdirectory(TestCases/test_arm_depthwise_conv_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_s8_opt)
add_subdirectory(TestCases/test_arm_ds_cnn_l_s8)
add_subdirectory(TestCases/test_arm_ds_cnn_s_s8)
add_subdirectory(TestCases/test_arm_elementwise_add_s16)
add_subdirectory(TestCases/test_arm_elementwise_sub_s16)
add_subdirectory(TestCases/test_arm_elementwise_add_s8)
add_subdirectory(TestCases/test_arm_elementwise_sub_s8)
add_subdirectory(TestCases/test_arm_argminmax_s8)
add_subdirectory(TestCases/test_arm_elementwise_mul_s16)
add_subdirectory(TestCases/test_arm_elementwise_mul_s8)
add_subdirectory(TestCases/test_arm_fully_connected_s16)
add_subdirectory(TestCases/test_arm_fully_connected_s8)
add_subdirectory(TestCases/test_arm_fully_connected_s4)
add_subdirectory(TestCases/test_arm_grouped_convolve_s8)
add_subdirectory(TestCases/test_arm_lstm_unidirectional_s8)
add_subdirectory(TestCases/test_arm_lstm_unidirectional_s16)
add_subdirectory(TestCases/test_arm_maximum_minimum_s8)
add_subdirectory(TestCases/test_arm_maximum_minimum_s16)
add_subdirectory(TestCases/test_arm_comparison_s8)
add_subdirectory(TestCases/test_arm_comparison_s16)
add_subdirectory(TestCases/test_arm_max_pool_s16)
add_subdirectory(TestCases/test_arm_max_pool_s8)
add_subdirectory(TestCases/test_arm_pad_s8)
add_subdirectory(TestCases/test_arm_pad_s16)
add_subdirectory(TestCases/test_arm_relu_s8)
add_subdirectory(TestCases/test_arm_relu_s16)
add_subdirectory(TestCases/test_arm_reverse_transpose_conv_s8)
add_subdirectory(TestCases/test_arm_softmax_s16)
add_subdirectory(TestCases/test_arm_softmax_s8)
add_subdirectory(TestCases/test_arm_softmax_s8_s16)
add_subdirectory(TestCases/test_arm_svdf_s8)
add_subdirectory(TestCases/test_arm_svdf_state_s16_s8)
add_subdirectory(TestCases/test_arm_transpose_conv_s8)
add_subdirectory(TestCases/test_arm_transpose_s8)
add_subdirectory(TestCases/test_arm_transpose_s16)
add_subdirectory(TestCases/test_arm_leaky_relu_s8)
add_subdirectory(TestCases/test_arm_prelu_s8)
add_subdirectory(TestCases/test_arm_prelu_s16)
add_subdirectory(TestCases/test_arm_quantize_f32_s16)
add_subdirectory(TestCases/test_arm_quantize_f32_s8)
add_subdirectory(TestCases/test_arm_quantize_s8_s8)
add_subdirectory(TestCases/test_arm_quantize_s16_s16)
add_subdirectory(TestCases/test_arm_dequantize_s8_f32)
add_subdirectory(TestCases/test_arm_dequantize_s16_f32)
add_subdirectory(TestCases/test_arm_concatenation_s32)
add_subdirectory(TestCases/test_arm_strided_slice_s8)
add_subdirectory(TestCases/test_arm_strided_slice_s16)
add_subdirectory(TestCases/test_arm_strided_slice_s32)
add_subdirectory(TestCases/test_arm_gather_s8)
add_subdirectory(TestCases/test_arm_gather_s16)
add_subdirectory(TestCases/test_arm_gather_nd_s8)
add_subdirectory(TestCases/test_arm_gather_nd_s16)
add_subdirectory(TestCases/test_arm_add_s8)
add_subdirectory(TestCases/test_arm_sub_s8)
add_subdirectory(TestCases/test_arm_mul_s8)
add_subdirectory(TestCases/test_arm_squared_difference_s8)
add_subdirectory(TestCases/test_arm_squared_difference_s16)
add_subdirectory(TestCases/test_arm_add_s16)
add_subdirectory(TestCases/test_arm_sub_s16)
add_subdirectory(TestCases/test_arm_mul_s16)
add_subdirectory(TestCases/test_arm_mean_s8)
add_subdirectory(TestCases/test_arm_mean_s16)
add_subdirectory(TestCases/test_arm_reduce_max_s8)
add_subdirectory(TestCases/test_arm_reduce_max_s16)
add_subdirectory(TestCases/test_arm_reduce_min_s8)
add_subdirectory(TestCases/test_arm_reduce_min_s16)
add_subdirectory(TestCases/test_arm_argminmax_s16)
add_subdirectory(TestCases/test_arm_hard_swish_s8)
add_subdirectory(TestCases/test_arm_hard_swish_s16)
add_subdirectory(TestCases/test_arm_rsqrt_s16)
add_subdirectory(TestCases/test_arm_space_to_depth_s8)
add_subdirectory(TestCases/test_arm_depth_to_space_s8)
add_subdirectory(TestCases/test_arm_space_to_batch_nd_s8)
add_subdirectory(TestCases/test_arm_batch_to_space_nd_s8)
add_subdirectory(TestCases/test_arm_space_to_depth_s16)
add_subdirectory(TestCases/test_arm_depth_to_space_s16)
add_subdirectory(TestCases/test_arm_space_to_batch_nd_s16)
add_subdirectory(TestCases/test_arm_batch_to_space_nd_s16)
add_subdirectory(TestCases/test_arm_resize_nearest_neighbor_s8)
add_subdirectory(TestCases/test_arm_resize_nearest_neighbor_s16)
add_subdirectory(TestCases/test_arm_abs_s8)
add_subdirectory(TestCases/test_arm_abs_s16)
add_subdirectory(TestCases/test_arm_sqrt_s8)
add_subdirectory(TestCases/test_arm_sqrt_s16)
add_subdirectory(TestCases/test_arm_broadcast_to_s8)
add_subdirectory(TestCases/test_arm_broadcast_to_s16)
add_subdirectory(TestCases/test_arm_dynamic_update_slice_s8)
add_subdirectory(TestCases/test_arm_dynamic_update_slice_s16)
add_subdirectory(TestCases/test_arm_mirror_pad_s8)
add_subdirectory(TestCases/test_arm_mirror_pad_s16)
add_subdirectory(TestCases/test_arm_reverse_sequence_s8)
add_subdirectory(TestCases/test_arm_reverse_sequence_s16)
add_subdirectory(TestCases/test_arm_scatter_nd_s8)
add_subdirectory(TestCases/test_arm_scatter_nd_s16)
add_subdirectory(TestCases/test_arm_select_v2_s8)
add_subdirectory(TestCases/test_arm_select_v2_s16)
add_subdirectory(TestCases/test_arm_tile_s8)
add_subdirectory(TestCases/test_arm_tile_s16)
add_subdirectory(TestCases/test_arm_where_s8)
add_subdirectory(TestCases/test_arm_where_s16)

if(ARM_NN_ENABLE_F32)
    add_subdirectory(TestCases/test_arm_convolve_packed_f32)
    add_subdirectory(TestCases/test_arm_nn_abs_f32)
    add_subdirectory(TestCases/test_arm_batch_matmul_f32)
    add_subdirectory(TestCases/test_arm_nn_activation_helpers_f32)
    add_subdirectory(TestCases/test_arm_elementwise_add_f32)
    add_subdirectory(TestCases/test_arm_elementwise_mul_f32)
    add_subdirectory(TestCases/test_arm_elementwise_sub_f32)
    add_subdirectory(TestCases/test_arm_gru_unidirectional_f32)
    add_subdirectory(TestCases/test_arm_lstm_unidirectional_f32)
    add_subdirectory(TestCases/test_arm_maximum_minimum_f32)
    add_subdirectory(TestCases/test_arm_nn_lstm_step_f32)
    add_subdirectory(TestCases/test_arm_nn_mean_f32)
    add_subdirectory(TestCases/test_arm_prelu_f32)
    add_subdirectory(TestCases/test_arm_reduce_sum_f32)
    add_subdirectory(TestCases/test_arm_reshape_f32)
    add_subdirectory(TestCases/test_arm_strided_slice_f32)
    add_subdirectory(TestCases/test_arm_transpose_conv_f32)
endif()

if(ARM_NN_ENABLE_F16)
    add_subdirectory(TestCases/test_arm_nn_abs_f16)
    add_subdirectory(TestCases/test_arm_batch_matmul_f16)
    add_subdirectory(TestCases/test_arm_convolve_packed_f16)
    add_subdirectory(TestCases/test_arm_elementwise_sub_f16)
    add_subdirectory(TestCases/test_arm_gru_unidirectional_f16)
    add_subdirectory(TestCases/test_arm_lstm_unidirectional_f16)
    add_subdirectory(TestCases/test_arm_maximum_minimum_f16)
    add_subdirectory(TestCases/test_arm_nn_lstm_step_f16)
    add_subdirectory(TestCases/test_arm_prelu_f16)
    add_subdirectory(TestCases/test_arm_reduce_sum_f16)
    add_subdirectory(TestCases/test_arm_reshape_f16)
    add_subdirectory(TestCases/test_arm_split_f16)
    add_subdirectory(TestCases/test_arm_strided_slice_f16)
    add_subdirectory(TestCases/test_arm_transpose_conv_f16)
endif()

# Host-only -Ofast NaN probe (#334 item d): rebuilds the six float elementwise
# kernels at -Ofast -ffinite-math-only -- the flag pair the harness suites
# deliberately never see (the directory-level -fno-finite-math-only above) --
# and fails on any clamp-to-bound of a NaN element. GNU-only because the host
# Clang leg swaps -Ofast out as deprecated (see the Clang branches above), and
# host-only because the FVP flow runs Unity .elf images, which this is not. The
# directory name carries no test_arm prefix on purpose: unittest_targets.py
# treats every TestCases/*test_arm* directory as a Unity suite and requires a
# Unity/ subdirectory, which this standalone probe does not have.
if(NOT BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300
   AND CMAKE_C_COMPILER_ID STREQUAL "GNU"
   AND ARM_NN_ENABLE_F32
   AND ARM_NN_ENABLE_F16)
    add_subdirectory(TestCases/elementwise_nan_ofast_probe)
endif()

# Host-only -Ofast NaN probe for the activation legs (#382). Same guard shape and
# same rationale as the elementwise probe above; it rebuilds arm_nn_activation_f32
# and arm_nn_activation_f16 under the hostile flag pair and fails if a NaN input
# comes back as 0.0 (RELU) or a clamp bound (RELU6). The directory name carries no
# test_arm prefix for the same unittest_targets.py reason given above. Only the
# scalar legs run on the host; the MVE RELU/RELU6 legs are covered on target.
if(NOT BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300
   AND CMAKE_C_COMPILER_ID STREQUAL "GNU"
   AND ARM_NN_ENABLE_F32
   AND ARM_NN_ENABLE_F16)
    add_subdirectory(TestCases/activation_nan_ofast_probe)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(MAKE_CMD "${Python3_EXECUTABLE}")
set(MAKE_CMD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/unittest_targets.py")
set(MAKE_CMD_SCRIPT_OPTION "--download-and-generate-test-runners")
MESSAGE(STATUS "Downloading Unity and generating test runners for CMSIS-NN unit tests if needed..")
execute_process(COMMAND ${MAKE_CMD} ${MAKE_CMD_SCRIPT} ${MAKE_CMD_SCRIPT_OPTION}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Allow vendored Unity CMake compatibility with CMake 4" FORCE)
add_subdirectory(Unity)

# Link common dependencies.
get_property(executables GLOBAL PROPERTY cmsis_nn_unit_test_executables)
string(REPLACE " " ";" cmsis_nn_unit_test_list_of_executables ${executables})
foreach(target ${cmsis_nn_unit_test_list_of_executables})
    target_link_libraries(${target} LINK_PUBLIC unity)
    target_link_libraries(${target} LINK_PUBLIC cmsis-nn)
    if(NOT BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
        target_link_libraries(${target} PRIVATE m)
    endif()
endforeach()

if(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
    add_library(retarget STATIC
        ${FVP_CORSTONE_300_PATH}/retarget.c
        ${FVP_CORSTONE_300_PATH}/uart.c)

    # Build CMSIS startup dependencies based on TARGET_CPU.
    string(REGEX REPLACE "^cortex-m([0-9]+)$" "ARMCM\\1" ARM_CPU ${CMAKE_SYSTEM_PROCESSOR})
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m33")
        set(ARM_FEATURES "_DSP_FP")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m4")
        set(ARM_FEATURES "_FP")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m7")
        set(ARM_FEATURES "_DP")
    else()
        set(ARM_FEATURES "")
    endif()
    add_library(cmsis_startup STATIC)
    target_sources(cmsis_startup PRIVATE
        ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/startup_${ARM_CPU}.c
        ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/system_${ARM_CPU}.c)
    target_include_directories(cmsis_startup PUBLIC
        ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include
        ${CMSIS_PATH}/CMSIS/Core/Include)
    target_compile_options(cmsis_startup INTERFACE -include${ARM_CPU}${ARM_FEATURES}.h)
    target_compile_definitions(cmsis_startup PRIVATE ${ARM_CPU}${ARM_FEATURES})

    # Linker file settings.
    set(LINK_FILE "${FVP_CORSTONE_300_PATH}/linker" CACHE PATH "Linker file.")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "ARMClang")
        set(LINK_FILE "${FVP_CORSTONE_300_PATH}/linker.scatter")
        set(LINK_FILE_OPTION "--scatter")
        set(LINK_ENTRY_OPTION "--entry")
        set(LINK_ENTRY "Reset_Handler")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(LINK_FILE "${FVP_CORSTONE_300_PATH}/linker.ld")
        set(LINK_FILE_OPTION "-T")
        set(LINK_ENTRY_OPTION "")
        set(LINK_ENTRY "")
    endif()

    # Link in FVP dependencies to every unit test.
    get_property(executables GLOBAL PROPERTY cmsis_nn_unit_test_executables)
    string(REPLACE " " ";" cmsis_nn_unit_test_list_of_executables ${executables})
    foreach(target ${cmsis_nn_unit_test_list_of_executables})
        target_link_libraries(${target} PRIVATE retarget)
        target_link_libraries(${target} PRIVATE $<TARGET_OBJECTS:cmsis_startup> cmsis_startup)
        add_dependencies(${target} retarget cmsis_startup)

        target_compile_definitions(${target} PUBLIC USING_FVP_CORSTONE_300)

        target_link_options(${target} PRIVATE ${LINK_FILE_OPTION} ${LINK_FILE} ${LINK_ENTRY_OPTION} ${LINK_ENTRY})
        set_target_properties(${target} PROPERTIES LINK_DEPENDS ${LINK_FILE})
    endforeach()
endif()
