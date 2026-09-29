# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

# Recompile the affected production objects: instrumenting only the probe would
# leave the library's shift-base exclusion in force. Executable definitions win
# over archive members; cmsis-nn supplies configuration and conversion helpers.
add_executable(test_arm_int4_decode_probe
    int4_decode_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ConvolutionFunctions/arm_depthwise_conv_s4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ConvolutionFunctions/arm_convolve_s4.c)
target_compile_options(test_arm_int4_decode_probe PRIVATE
    -fsanitize=address,undefined,shift-base -fno-sanitize-recover=all -fno-omit-frame-pointer)
target_link_options(test_arm_int4_decode_probe PRIVATE
    -fsanitize=address,undefined,shift-base -fno-sanitize-recover=all)
target_link_libraries(test_arm_int4_decode_probe PRIVATE cmsis-nn m)
add_test(NAME test_arm_int4_decode_probe COMMAND test_arm_int4_decode_probe)
