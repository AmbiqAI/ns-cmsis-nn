add_cmsis_nn_unit_test_executable(test_arm_rsqrt_s16)

target_sources(test_arm_rsqrt_s16 PRIVATE
    Unity/unity_test_arm_rsqrt_s16.c
    Unity/TestRunner/unity_test_arm_rsqrt_s16_runner.c)
