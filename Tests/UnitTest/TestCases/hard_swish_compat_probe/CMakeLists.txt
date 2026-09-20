# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

# Instrument the actual kernel as well as the inline helpers. The library's
# global shift-base exclusion must not hide this compatibility regression.
add_executable(test_arm_hard_swish_compat_probe
    hard_swish_compat_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ActivationFunctions/arm_hard_swish_compat_s8.c)
target_compile_options(test_arm_hard_swish_compat_probe PRIVATE
    -fsanitize=address,undefined,shift-base -fno-sanitize-recover=all -fno-omit-frame-pointer)
target_link_options(test_arm_hard_swish_compat_probe PRIVATE
    -fsanitize=address,undefined,shift-base -fno-sanitize-recover=all)
target_link_libraries(test_arm_hard_swish_compat_probe PRIVATE cmsis-nn m)
add_test(NAME test_arm_hard_swish_compat_probe COMMAND test_arm_hard_swish_compat_probe)
