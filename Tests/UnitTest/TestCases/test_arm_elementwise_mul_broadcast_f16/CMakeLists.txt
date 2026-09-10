#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

add_cmsis_nn_unit_test_executable(test_arm_elementwise_mul_broadcast_f16)

target_sources(test_arm_elementwise_mul_broadcast_f16 PRIVATE
    Unity/unity_test_arm_elementwise_mul_broadcast_f16.c
    Unity/TestRunner/unity_test_arm_elementwise_mul_broadcast_f16_runner.c)

# The FVP images keep the template's default of 2000 shape pairs; the host build (ctest,
# sanitizers) draws 20000, being the cheap leg. EW_BCAST_FUZZ_ITERS on the cmake command line
# overrides both, e.g. for a long QEMU run (#415).
if(EW_BCAST_FUZZ_ITERS)
    target_compile_definitions(test_arm_elementwise_mul_broadcast_f16 PRIVATE EW_BCAST_FUZZ_ITERS=${EW_BCAST_FUZZ_ITERS})
elseif(NOT BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
    target_compile_definitions(test_arm_elementwise_mul_broadcast_f16 PRIVATE EW_BCAST_FUZZ_ITERS=20000)
endif()
