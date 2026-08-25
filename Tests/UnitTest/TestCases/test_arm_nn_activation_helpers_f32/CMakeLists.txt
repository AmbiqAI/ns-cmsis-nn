#
# Copyright (C) 2026 Arm Limited or its affiliates.
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

add_cmsis_nn_unit_test_executable(test_arm_nn_activation_helpers_f32)

target_sources(test_arm_nn_activation_helpers_f32 PRIVATE
    Unity/unity_test_arm_nn_activation_helpers_f32.c
    Unity/TestRunner/unity_test_arm_nn_activation_helpers_f32_runner.c)

# This suite asserts on NaN inputs, so it must be built with conforming IEEE
# semantics. The default library build sets CMSIS_OPTIMIZATION_LEVEL to -Ofast,
# which implies -ffinite-math-only: the compiler then deletes the `ax != ax`
# guard in arm_nn_tanh_scalar_ref_f32() and reassociates the softmax domain
# clamp, and the NaN assertions here become false. Pin the flag on the target
# rather than relying on whatever the enclosing build happens to pass.
target_compile_options(test_arm_nn_activation_helpers_f32 PRIVATE -fno-finite-math-only)
