# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

# Large outputs exercise both accumulator widths without using the Corstone heap.
add_executable(test_resize_bilinear_wide_probe
    resize_bilinear_wide_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ReshapeFunctions/arm_resize_bilinear_s8.c)
target_compile_options(test_resize_bilinear_wide_probe PRIVATE
    -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
target_link_options(test_resize_bilinear_wide_probe PRIVATE
    -fsanitize=address,undefined -fno-sanitize-recover=all)
target_link_libraries(test_resize_bilinear_wide_probe PRIVATE cmsis-nn)
add_test(NAME test_resize_bilinear_wide_probe COMMAND test_resize_bilinear_wide_probe)
