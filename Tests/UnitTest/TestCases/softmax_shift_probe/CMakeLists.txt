# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

# Recompile the softmax objects with shift-base checks, which the host sanitizer leg masks (#704). Executable
# definitions win over archive members; cmsis-nn supplies the remaining helpers.
add_executable(test_softmax_shift_probe
    softmax_shift_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/SoftmaxFunctions/arm_softmax_s8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/SoftmaxFunctions/arm_softmax_s8_s16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/SoftmaxFunctions/arm_softmax_u8.c)
target_compile_options(test_softmax_shift_probe PRIVATE
    -fsanitize=address,undefined,shift-base -fno-sanitize-recover=all -fno-omit-frame-pointer)
target_link_options(test_softmax_shift_probe PRIVATE
    -fsanitize=address,undefined,shift-base -fno-sanitize-recover=all)
target_link_libraries(test_softmax_shift_probe PRIVATE cmsis-nn m)
add_test(NAME test_softmax_shift_probe COMMAND test_softmax_shift_probe)
