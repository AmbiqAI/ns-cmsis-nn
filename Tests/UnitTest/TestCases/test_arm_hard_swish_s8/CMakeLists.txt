#
# SPDX-FileCopyrightText: Copyright 2025 Ambiq
# SPDX-License-Identifier: Apache-2.0
#

# Hard-swish unit test target
add_cmsis_nn_unit_test_executable(test_arm_hard_swish_s8)

# Unity test + runner
target_sources(test_arm_hard_swish_s8 PRIVATE
  Unity/unity_test_arm_hard_swish_s8.c
  Unity/TestRunner/unity_test_arm_hard_swish_s8_runner.c
)

# Collect only headers under ../TestData/hard_swish_*/*
file(GLOB_RECURSE HS_DATA_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../TestData/hard_swish_*/*.h"
)

# (Optional) add them as sources so IDEs show them
target_sources(test_arm_hard_swish_s8 PRIVATE ${HS_DATA_FILES})

# Force recompilation of the Unity TU when any test-data header changes
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/Unity/unity_test_arm_hard_swish_s8.c"
  PROPERTIES OBJECT_DEPENDS "${HS_DATA_FILES}"
)

# The runner usually doesn't include test data, but it's cheap to add the dependency too
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/Unity/TestRunner/unity_test_arm_hard_swish_s8_runner.c"
  PROPERTIES OBJECT_DEPENDS "${HS_DATA_FILES}"
)

# Enable compiler-generated dependency files for transitive includes
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_arm_hard_swish_s8 PRIVATE -MMD -MP)
endif()

# If you prefer cleaner includes (e.g., "TestData/.../test_data.h"), you can also add:
# target_include_directories(test_arm_hard_swish_s8 PRIVATE
#   "${CMAKE_CURRENT_SOURCE_DIR}/.."
# )
