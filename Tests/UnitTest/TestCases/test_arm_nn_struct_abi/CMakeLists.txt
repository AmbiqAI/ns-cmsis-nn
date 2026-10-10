# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

add_cmsis_nn_unit_test_executable(test_arm_nn_struct_abi)
target_sources(test_arm_nn_struct_abi PRIVATE
    Unity/unity_test_arm_nn_struct_abi.c
    Unity/TestRunner/unity_test_arm_nn_struct_abi_runner.c)
# The caller side uses short enums whatever the library's convention (#764).
target_compile_options(test_arm_nn_struct_abi PRIVATE -fshort-enums)
