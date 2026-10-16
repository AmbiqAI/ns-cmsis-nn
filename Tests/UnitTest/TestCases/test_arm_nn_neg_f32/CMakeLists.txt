add_cmsis_nn_unit_test_executable(test_arm_nn_neg_f32)

target_sources(test_arm_nn_neg_f32 PRIVATE
    Unity/unity_test_arm_nn_neg_f32.c
    Unity/TestRunner/unity_test_arm_nn_neg_f32_runner.c)
