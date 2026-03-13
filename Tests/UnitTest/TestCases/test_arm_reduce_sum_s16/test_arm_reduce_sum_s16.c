#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../Utils/validate.h"

void reduce_sum_s16(void)
{
    #include "../TestData/reduce_sum_s16_test/test_data.h"
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_SUM_S16_TEST_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_SUM_S16_TEST_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_SUM_S16_TEST_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_SUM_S16_TEST_OUT_DIM;

    const int16_t *input_data = reduce_sum_s16_test_input_tensor;
    const int16_t *const output_ref = reduce_sum_s16_test_output;
    const int32_t output_ref_size = REDUCE_SUM_S16_TEST_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_sum_s16(
        input_data,
        &input_dims,
        REDUCE_SUM_S16_TEST_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        REDUCE_SUM_S16_TEST_OUTPUT_OFFSET,
        REDUCE_SUM_S16_TEST_OUTPUT_MULTIPLIER,
        REDUCE_SUM_S16_TEST_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void reduce_sum_s16_axis_2_3(void)
{
    #include "../TestData/reduce_sum_s16_test_axis_2_3/test_data.h"
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_SUM_S16_TEST_AXIS_2_3_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_SUM_S16_TEST_AXIS_2_3_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_SUM_S16_TEST_AXIS_2_3_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_SUM_S16_TEST_AXIS_2_3_OUT_DIM;

    const int16_t *input_data = reduce_sum_s16_test_axis_2_3_input_tensor;
    const int16_t *const output_ref = reduce_sum_s16_test_axis_2_3_output;
    const int32_t output_ref_size = REDUCE_SUM_S16_TEST_AXIS_2_3_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_sum_s16(
        input_data,
        &input_dims,
        REDUCE_SUM_S16_TEST_AXIS_2_3_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        REDUCE_SUM_S16_TEST_AXIS_2_3_OUTPUT_OFFSET,
        REDUCE_SUM_S16_TEST_AXIS_2_3_OUTPUT_MULTIPLIER,
        REDUCE_SUM_S16_TEST_AXIS_2_3_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}