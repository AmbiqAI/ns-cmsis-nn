add_cmsis_nn_unit_test_executable(test_arm_int32_predicates)
target_sources(test_arm_int32_predicates PRIVATE
    Unity/unity_test_arm_int32_predicates.c
    Unity/TestRunner/unity_test_arm_int32_predicates_runner.c)
