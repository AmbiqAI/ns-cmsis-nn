# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

add_executable(test_arm_transpose_skipped_rows_probe transpose_skipped_rows_probe.c)
target_link_libraries(test_arm_transpose_skipped_rows_probe PRIVATE cmsis-nn m)
# Observe entry into the actual row helper: output parity cannot detect an
# unused out-of-bounds filter pointer formed by a wholly skipped row.
target_link_options(test_arm_transpose_skipped_rows_probe PRIVATE
    "LINKER:--wrap=arm_nn_transpose_conv_row_s8_s32")
add_test(NAME test_arm_transpose_skipped_rows_probe COMMAND test_arm_transpose_skipped_rows_probe)
