#
# Copyright (C) 2022 Arm Limited or its affiliates.
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

add_cmsis_nn_unit_test_executable(test_arm_elementwise_add_s16)

target_sources(test_arm_elementwise_add_s16 PRIVATE
    Unity/unity_test_arm_elementwise_add_s16.c
    Unity/TestRunner/unity_test_arm_elementwise_add_s16_runner.c)

# Compile the header helpers in both rounding modes, independently of the library's mode.
foreach(rounding default single)
    set(target test_arm_requantize_shifted_mve_${rounding})
    add_cmsis_nn_unit_test_executable(${target})
    target_sources(${target} PRIVATE test_arm_requantize_shifted_mve.c)
    if(rounding STREQUAL "single")
        target_compile_definitions(${target} PRIVATE TEST_REQUANTIZE_SINGLE_ROUNDING)
    endif()
endforeach()
