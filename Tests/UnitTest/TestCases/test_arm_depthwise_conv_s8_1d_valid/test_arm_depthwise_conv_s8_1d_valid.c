#include <arm_nnfunctions.h>
#include <unity.h>
#include "../TestData/basic_1d_valid_40ch_30w/test_data.h"
#include "../TestData/edge_equal_k_1d_valid_33ch_3w/test_data.h"
#include "../TestData/k1_1d_valid_40ch_31w/test_data.h"
#include "../TestData/k3_1d_valid_40ch_29w/test_data.h"
#include "../TestData/k5_1d_valid_40ch_64w/test_data.h"
#include "../TestData/k7_1d_valid_37ch_127w/test_data.h"
#include "../TestData/wide_k3_1d_valid_35ch_255w/test_data.h"
#include "../TestData/wide_k5_1d_valid_35ch_256h/test_data.h"
#include "../TestData/k7_1d_valid_16ch_32h/test_data.h"
#include "../TestData/edge_equal_k_1d_valid_8ch_3h/test_data.h"
#include "../TestData/small_k3_1d_valid_12ch_16h/test_data.h"
#include "../TestData/medium_k5_1d_valid_20ch_80h/test_data.h"
#include "../TestData/large_k9_1d_valid_28ch_200h/test_data.h"
#include "../TestData/edge_equal_k_1d_valid_128ch_15w/test_data.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"




void basic_arm_depthwise_conv_s8_1d_valid_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_1D_VALID_40CH_30W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_1d_valid_40ch_30w_biases;
    const int8_t *kernel_data = basic_1d_valid_40ch_30w_weights;
    const int8_t *input_data = basic_1d_valid_40ch_30w_input_tensor;

    input_dims.n = BASIC_1D_VALID_40CH_30W_INPUT_BATCHES;
    input_dims.w = BASIC_1D_VALID_40CH_30W_INPUT_W;
    input_dims.h = BASIC_1D_VALID_40CH_30W_INPUT_H;
    input_dims.c = BASIC_1D_VALID_40CH_30W_IN_CH;
    filter_dims.w = BASIC_1D_VALID_40CH_30W_FILTER_X;
    filter_dims.h = BASIC_1D_VALID_40CH_30W_FILTER_Y;
    output_dims.w = BASIC_1D_VALID_40CH_30W_OUTPUT_W;
    output_dims.h = BASIC_1D_VALID_40CH_30W_OUTPUT_H;
    output_dims.c = BASIC_1D_VALID_40CH_30W_OUT_CH;

    dw_conv_params.padding.w = BASIC_1D_VALID_40CH_30W_PAD_X;
    dw_conv_params.padding.h = BASIC_1D_VALID_40CH_30W_PAD_Y;
    dw_conv_params.stride.w = BASIC_1D_VALID_40CH_30W_STRIDE_X;
    dw_conv_params.stride.h = BASIC_1D_VALID_40CH_30W_STRIDE_Y;
    dw_conv_params.dilation.w = BASIC_1D_VALID_40CH_30W_DILATION_X;
    dw_conv_params.dilation.h = BASIC_1D_VALID_40CH_30W_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = BASIC_1D_VALID_40CH_30W_INPUT_OFFSET;
    dw_conv_params.output_offset = BASIC_1D_VALID_40CH_30W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = BASIC_1D_VALID_40CH_30W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = BASIC_1D_VALID_40CH_30W_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_1d_valid_40ch_30w_output_mult;
    quant_params.shift = (int32_t *)basic_1d_valid_40ch_30w_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif




    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, basic_1d_valid_40ch_30w_output_ref, BASIC_1D_VALID_40CH_30W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, basic_1d_valid_40ch_30w_output_ref, BASIC_1D_VALID_40CH_30W_DST_SIZE));
}

void k1_1d_valid_40ch_31w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[K1_1D_VALID_40CH_31W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;


    const int32_t *bias_data = k1_1d_valid_40ch_31w_biases;
    const int8_t *kernel_data = k1_1d_valid_40ch_31w_weights;
    const int8_t *input_data = k1_1d_valid_40ch_31w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = K1_1D_VALID_40CH_31W_INPUT_BATCHES;
    input_dims.w = K1_1D_VALID_40CH_31W_INPUT_W;
    input_dims.h = K1_1D_VALID_40CH_31W_INPUT_H;
    input_dims.c = K1_1D_VALID_40CH_31W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = K1_1D_VALID_40CH_31W_FILTER_X;
    filter_dims.h = K1_1D_VALID_40CH_31W_FILTER_Y;
    filter_dims.c = K1_1D_VALID_40CH_31W_OUT_CH;

    // Set up output dimensions
    output_dims.w = K1_1D_VALID_40CH_31W_OUTPUT_W;
    output_dims.h = K1_1D_VALID_40CH_31W_OUTPUT_H;
    output_dims.c = K1_1D_VALID_40CH_31W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = K1_1D_VALID_40CH_31W_PAD_X;
    dw_conv_params.padding.h = K1_1D_VALID_40CH_31W_PAD_Y;
    dw_conv_params.stride.w = K1_1D_VALID_40CH_31W_STRIDE_X;
    dw_conv_params.stride.h = K1_1D_VALID_40CH_31W_STRIDE_Y;
    dw_conv_params.dilation.w = K1_1D_VALID_40CH_31W_DILATION_X;
    dw_conv_params.dilation.h = K1_1D_VALID_40CH_31W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = K1_1D_VALID_40CH_31W_INPUT_OFFSET;
    dw_conv_params.output_offset = K1_1D_VALID_40CH_31W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = K1_1D_VALID_40CH_31W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = K1_1D_VALID_40CH_31W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)k1_1d_valid_40ch_31w_output_mult;
    quant_params.shift = (int32_t *)k1_1d_valid_40ch_31w_output_shift;
    
    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    #if defined(ARM_MATH_DSP)
        TEST_ASSERT_TRUE(ctx.size > 0);
    #else
        TEST_ASSERT_EQUAL(ctx.size, 0);
    #endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                                bias_data);

    // Test the 1D depthwise convolution wrapper function
    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                                           &weights_sum_ctx,
                                                                           &dw_conv_params,
                                                                           &quant_params,
                                                                           &input_dims,
                                                                           input_data,
                                                                           &filter_dims,
                                                                           kernel_data,
                                                                           &bias_dims,
                                                                           bias_data,
                                                                           &output_dims,
                                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k1_1d_valid_40ch_31w_output_ref, K1_1D_VALID_40CH_31W_DST_SIZE));
    
    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k1_1d_valid_40ch_31w_output_ref, K1_1D_VALID_40CH_31W_DST_SIZE));
}

void k3_1d_valid_40ch_29w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[K3_1D_VALID_40CH_29W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(k3_1d_valid_40ch_29w_biases, K3_1D_VALID_40CH_29W_OUT_CH);
    const int8_t *kernel_data = k3_1d_valid_40ch_29w_weights;
    const int8_t *input_data = k3_1d_valid_40ch_29w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = K3_1D_VALID_40CH_29W_INPUT_BATCHES;
    input_dims.w = K3_1D_VALID_40CH_29W_INPUT_W;
    input_dims.h = K3_1D_VALID_40CH_29W_INPUT_H;
    input_dims.c = K3_1D_VALID_40CH_29W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = K3_1D_VALID_40CH_29W_FILTER_X;
    filter_dims.h = K3_1D_VALID_40CH_29W_FILTER_Y;
    filter_dims.c = K3_1D_VALID_40CH_29W_OUT_CH;

    // Set up output dimensions
    output_dims.w = K3_1D_VALID_40CH_29W_OUTPUT_W;
    output_dims.h = K3_1D_VALID_40CH_29W_OUTPUT_H;
    output_dims.c = K3_1D_VALID_40CH_29W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = K3_1D_VALID_40CH_29W_PAD_X;
    dw_conv_params.padding.h = K3_1D_VALID_40CH_29W_PAD_Y;
    dw_conv_params.stride.w = K3_1D_VALID_40CH_29W_STRIDE_X;
    dw_conv_params.stride.h = K3_1D_VALID_40CH_29W_STRIDE_Y;
    dw_conv_params.dilation.w = K3_1D_VALID_40CH_29W_DILATION_X;
    dw_conv_params.dilation.h = K3_1D_VALID_40CH_29W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = K3_1D_VALID_40CH_29W_INPUT_OFFSET;
    dw_conv_params.output_offset = K3_1D_VALID_40CH_29W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = K3_1D_VALID_40CH_29W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = K3_1D_VALID_40CH_29W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)k3_1d_valid_40ch_29w_output_mult;
    quant_params.shift = (int32_t *)k3_1d_valid_40ch_29w_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif




    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k3_1d_valid_40ch_29w_output_ref, K3_1D_VALID_40CH_29W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k3_1d_valid_40ch_29w_output_ref, K3_1D_VALID_40CH_29W_DST_SIZE));
}

void k5_1d_valid_40ch_64w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[K5_1D_VALID_40CH_64W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(k5_1d_valid_40ch_64w_biases, K5_1D_VALID_40CH_64W_OUT_CH);
    const int8_t *kernel_data = k5_1d_valid_40ch_64w_weights;
    const int8_t *input_data = k5_1d_valid_40ch_64w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = K5_1D_VALID_40CH_64W_INPUT_BATCHES;
    input_dims.w = K5_1D_VALID_40CH_64W_INPUT_W;
    input_dims.h = K5_1D_VALID_40CH_64W_INPUT_H;
    input_dims.c = K5_1D_VALID_40CH_64W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = K5_1D_VALID_40CH_64W_FILTER_X;
    filter_dims.h = K5_1D_VALID_40CH_64W_FILTER_Y;
    filter_dims.c = K5_1D_VALID_40CH_64W_OUT_CH;

    // Set up output dimensions
    output_dims.w = K5_1D_VALID_40CH_64W_OUTPUT_W;
    output_dims.h = K5_1D_VALID_40CH_64W_OUTPUT_H;
    output_dims.c = K5_1D_VALID_40CH_64W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = K5_1D_VALID_40CH_64W_PAD_X;
    dw_conv_params.padding.h = K5_1D_VALID_40CH_64W_PAD_Y;
    dw_conv_params.stride.w = K5_1D_VALID_40CH_64W_STRIDE_X;
    dw_conv_params.stride.h = K5_1D_VALID_40CH_64W_STRIDE_Y;
    dw_conv_params.dilation.w = K5_1D_VALID_40CH_64W_DILATION_X;
    dw_conv_params.dilation.h = K5_1D_VALID_40CH_64W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = K5_1D_VALID_40CH_64W_INPUT_OFFSET;
    dw_conv_params.output_offset = K5_1D_VALID_40CH_64W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = K5_1D_VALID_40CH_64W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = K5_1D_VALID_40CH_64W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)k5_1d_valid_40ch_64w_output_mult;
    quant_params.shift = (int32_t *)k5_1d_valid_40ch_64w_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif




    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k5_1d_valid_40ch_64w_output_ref, K5_1D_VALID_40CH_64W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k5_1d_valid_40ch_64w_output_ref, K5_1D_VALID_40CH_64W_DST_SIZE));
}

void k7_1d_valid_37ch_127w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[K7_1D_VALID_37CH_127W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(k7_1d_valid_37ch_127w_biases, K7_1D_VALID_37CH_127W_OUT_CH);
    const int8_t *kernel_data = k7_1d_valid_37ch_127w_weights;
    const int8_t *input_data = k7_1d_valid_37ch_127w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = K7_1D_VALID_37CH_127W_INPUT_BATCHES;
    input_dims.w = K7_1D_VALID_37CH_127W_INPUT_W;
    input_dims.h = K7_1D_VALID_37CH_127W_INPUT_H;
    input_dims.c = K7_1D_VALID_37CH_127W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = K7_1D_VALID_37CH_127W_FILTER_X;
    filter_dims.h = K7_1D_VALID_37CH_127W_FILTER_Y;
    filter_dims.c = K7_1D_VALID_37CH_127W_OUT_CH;

    // Set up output dimensions
    output_dims.w = K7_1D_VALID_37CH_127W_OUTPUT_W;
    output_dims.h = K7_1D_VALID_37CH_127W_OUTPUT_H;
    output_dims.c = K7_1D_VALID_37CH_127W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = K7_1D_VALID_37CH_127W_PAD_X;
    dw_conv_params.padding.h = K7_1D_VALID_37CH_127W_PAD_Y;
    dw_conv_params.stride.w = K7_1D_VALID_37CH_127W_STRIDE_X;
    dw_conv_params.stride.h = K7_1D_VALID_37CH_127W_STRIDE_Y;
    dw_conv_params.dilation.w = K7_1D_VALID_37CH_127W_DILATION_X;
    dw_conv_params.dilation.h = K7_1D_VALID_37CH_127W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = K7_1D_VALID_37CH_127W_INPUT_OFFSET;
    dw_conv_params.output_offset = K7_1D_VALID_37CH_127W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = K7_1D_VALID_37CH_127W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = K7_1D_VALID_37CH_127W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)k7_1d_valid_37ch_127w_output_mult;
    quant_params.shift = (int32_t *)k7_1d_valid_37ch_127w_output_shift;


    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif




    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k7_1d_valid_37ch_127w_output_ref, K7_1D_VALID_37CH_127W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k7_1d_valid_37ch_127w_output_ref, K7_1D_VALID_37CH_127W_DST_SIZE));
}

void edge_equal_k_1d_valid_33ch_3w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[EDGE_EQUAL_K_1D_VALID_33CH_3W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(edge_equal_k_1d_valid_33ch_3w_biases, EDGE_EQUAL_K_1D_VALID_33CH_3W_OUT_CH);
    const int8_t *kernel_data = edge_equal_k_1d_valid_33ch_3w_weights;
    const int8_t *input_data = edge_equal_k_1d_valid_33ch_3w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = EDGE_EQUAL_K_1D_VALID_33CH_3W_INPUT_BATCHES;
    input_dims.w = EDGE_EQUAL_K_1D_VALID_33CH_3W_INPUT_W;
    input_dims.h = EDGE_EQUAL_K_1D_VALID_33CH_3W_INPUT_H;
    input_dims.c = EDGE_EQUAL_K_1D_VALID_33CH_3W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = EDGE_EQUAL_K_1D_VALID_33CH_3W_FILTER_X;
    filter_dims.h = EDGE_EQUAL_K_1D_VALID_33CH_3W_FILTER_Y;
    filter_dims.c = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUT_CH;

    // Set up output dimensions
    output_dims.w = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUTPUT_W;
    output_dims.h = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUTPUT_H;
    output_dims.c = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = EDGE_EQUAL_K_1D_VALID_33CH_3W_PAD_X;
    dw_conv_params.padding.h = EDGE_EQUAL_K_1D_VALID_33CH_3W_PAD_Y;
    dw_conv_params.stride.w = EDGE_EQUAL_K_1D_VALID_33CH_3W_STRIDE_X;
    dw_conv_params.stride.h = EDGE_EQUAL_K_1D_VALID_33CH_3W_STRIDE_Y;
    dw_conv_params.dilation.w = EDGE_EQUAL_K_1D_VALID_33CH_3W_DILATION_X;
    dw_conv_params.dilation.h = EDGE_EQUAL_K_1D_VALID_33CH_3W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = EDGE_EQUAL_K_1D_VALID_33CH_3W_INPUT_OFFSET;
    dw_conv_params.output_offset = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = EDGE_EQUAL_K_1D_VALID_33CH_3W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)edge_equal_k_1d_valid_33ch_3w_output_mult;
    quant_params.shift = (int32_t *)edge_equal_k_1d_valid_33ch_3w_output_shift;


    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif




    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, edge_equal_k_1d_valid_33ch_3w_output_ref, EDGE_EQUAL_K_1D_VALID_33CH_3W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, edge_equal_k_1d_valid_33ch_3w_output_ref, EDGE_EQUAL_K_1D_VALID_33CH_3W_DST_SIZE));
}

void wide_k3_1d_valid_35ch_255w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[WIDE_K3_1D_VALID_35CH_255W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(wide_k3_1d_valid_35ch_255w_biases, WIDE_K3_1D_VALID_35CH_255W_OUT_CH);
    const int8_t *kernel_data = wide_k3_1d_valid_35ch_255w_weights;
    const int8_t *input_data = wide_k3_1d_valid_35ch_255w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = WIDE_K3_1D_VALID_35CH_255W_INPUT_BATCHES;
    input_dims.w = WIDE_K3_1D_VALID_35CH_255W_INPUT_W;
    input_dims.h = WIDE_K3_1D_VALID_35CH_255W_INPUT_H;
    input_dims.c = WIDE_K3_1D_VALID_35CH_255W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = WIDE_K3_1D_VALID_35CH_255W_FILTER_X;
    filter_dims.h = WIDE_K3_1D_VALID_35CH_255W_FILTER_Y;
    filter_dims.c = WIDE_K3_1D_VALID_35CH_255W_OUT_CH;

    // Set up output dimensions
    output_dims.w = WIDE_K3_1D_VALID_35CH_255W_OUTPUT_W;
    output_dims.h = WIDE_K3_1D_VALID_35CH_255W_OUTPUT_H;
    output_dims.c = WIDE_K3_1D_VALID_35CH_255W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = WIDE_K3_1D_VALID_35CH_255W_PAD_X;
    dw_conv_params.padding.h = WIDE_K3_1D_VALID_35CH_255W_PAD_Y;
    dw_conv_params.stride.w = WIDE_K3_1D_VALID_35CH_255W_STRIDE_X;
    dw_conv_params.stride.h = WIDE_K3_1D_VALID_35CH_255W_STRIDE_Y;
    dw_conv_params.dilation.w = WIDE_K3_1D_VALID_35CH_255W_DILATION_X;
    dw_conv_params.dilation.h = WIDE_K3_1D_VALID_35CH_255W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = WIDE_K3_1D_VALID_35CH_255W_INPUT_OFFSET;
    dw_conv_params.output_offset = WIDE_K3_1D_VALID_35CH_255W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = WIDE_K3_1D_VALID_35CH_255W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = WIDE_K3_1D_VALID_35CH_255W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)wide_k3_1d_valid_35ch_255w_output_mult;
    quant_params.shift = (int32_t *)wide_k3_1d_valid_35ch_255w_output_shift;


    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif




    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, wide_k3_1d_valid_35ch_255w_output_ref, WIDE_K3_1D_VALID_35CH_255W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, wide_k3_1d_valid_35ch_255w_output_ref, WIDE_K3_1D_VALID_35CH_255W_DST_SIZE));
}


void wide_k5_1d_valid_35ch_256h_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[WIDE_K5_1D_VALID_35CH_256H_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(wide_k5_1d_valid_35ch_256h_biases, WIDE_K5_1D_VALID_35CH_256H_OUT_CH);
    const int8_t *kernel_data = wide_k5_1d_valid_35ch_256h_weights;
    const int8_t *input_data = wide_k5_1d_valid_35ch_256h_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = WIDE_K5_1D_VALID_35CH_256H_INPUT_BATCHES;
    input_dims.w = WIDE_K5_1D_VALID_35CH_256H_INPUT_W;
    input_dims.h = WIDE_K5_1D_VALID_35CH_256H_INPUT_H;
    input_dims.c = WIDE_K5_1D_VALID_35CH_256H_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = WIDE_K5_1D_VALID_35CH_256H_FILTER_X;
    filter_dims.h = WIDE_K5_1D_VALID_35CH_256H_FILTER_Y;
    filter_dims.c = WIDE_K5_1D_VALID_35CH_256H_OUT_CH;

    // Set up output dimensions
    output_dims.w = WIDE_K5_1D_VALID_35CH_256H_OUTPUT_W;
    output_dims.h = WIDE_K5_1D_VALID_35CH_256H_OUTPUT_H;
    output_dims.c = WIDE_K5_1D_VALID_35CH_256H_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = WIDE_K5_1D_VALID_35CH_256H_PAD_X;
    dw_conv_params.padding.h = WIDE_K5_1D_VALID_35CH_256H_PAD_Y;
    dw_conv_params.stride.w = WIDE_K5_1D_VALID_35CH_256H_STRIDE_X;
    dw_conv_params.stride.h = WIDE_K5_1D_VALID_35CH_256H_STRIDE_Y;
    dw_conv_params.dilation.w = WIDE_K5_1D_VALID_35CH_256H_DILATION_X;
    dw_conv_params.dilation.h = WIDE_K5_1D_VALID_35CH_256H_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = WIDE_K5_1D_VALID_35CH_256H_INPUT_OFFSET;
    dw_conv_params.output_offset = WIDE_K5_1D_VALID_35CH_256H_OUTPUT_OFFSET;
    dw_conv_params.activation.min = WIDE_K5_1D_VALID_35CH_256H_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = WIDE_K5_1D_VALID_35CH_256H_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)wide_k5_d1_valid_35ch_256h_output_mult;
    quant_params.shift = (int32_t *)wide_k5_1d_valid_35ch_256h_output_shift;


    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, wide_k5_1d_valid_35ch_256h_output_ref, WIDE_K5_1D_VALID_35CH_256H_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);


    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, wide_k5_1d_valid_35ch_256h_output_ref, WIDE_K5_1D_VALID_35CH_256H_DST_SIZE));
}

void k7_1d_valid_16ch_32h_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[K7_1D_VALID_16CH_32H_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = K7_1D_VALID_16CH_32H_DST_SIZE;
    const int32_t *bias_data = get_bias_address(k7_1d_valid_16ch_32h_biases, K7_1D_VALID_16CH_32H_OUT_CH);
    const int8_t *kernel_data = k7_1d_valid_16ch_32h_weights;
    const int8_t *input_data = k7_1d_valid_16ch_32h_input_tensor;

    // Set up input dimensions: 1D convolution with width=1
    input_dims.n = K7_1D_VALID_16CH_32H_INPUT_BATCHES;
    input_dims.w = K7_1D_VALID_16CH_32H_INPUT_W;
    input_dims.h = K7_1D_VALID_16CH_32H_INPUT_H;
    input_dims.c = K7_1D_VALID_16CH_32H_IN_CH;

    // Set up filter dimensions: 1D filter with width=1
    filter_dims.w = K7_1D_VALID_16CH_32H_FILTER_X;
    filter_dims.h = K7_1D_VALID_16CH_32H_FILTER_Y;
    filter_dims.c = K7_1D_VALID_16CH_32H_OUT_CH;

    // Set up output dimensions
    output_dims.w = K7_1D_VALID_16CH_32H_OUTPUT_W;
    output_dims.h = K7_1D_VALID_16CH_32H_OUTPUT_H;
    output_dims.c = K7_1D_VALID_16CH_32H_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = K7_1D_VALID_16CH_32H_PAD_X;
    dw_conv_params.padding.h = K7_1D_VALID_16CH_32H_PAD_Y;
    dw_conv_params.stride.w = K7_1D_VALID_16CH_32H_STRIDE_X;
    dw_conv_params.stride.h = K7_1D_VALID_16CH_32H_STRIDE_Y;
    dw_conv_params.dilation.w = K7_1D_VALID_16CH_32H_DILATION_X;
    dw_conv_params.dilation.h = K7_1D_VALID_16CH_32H_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = K7_1D_VALID_16CH_32H_INPUT_OFFSET;
    dw_conv_params.output_offset = K7_1D_VALID_16CH_32H_OUTPUT_OFFSET;
    dw_conv_params.activation.min = K7_1D_VALID_16CH_32H_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = K7_1D_VALID_16CH_32H_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)k7_1d_valid_16ch_32h_output_mult;
    quant_params.shift = (int32_t *)k7_1d_valid_16ch_32h_output_shift;
    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    // Test the 1D depthwise convolution wrapper function
    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                                           &weights_sum_ctx,
                                                                           &dw_conv_params,
                                                                           &quant_params,
                                                                           &input_dims,
                                                                           input_data,
                                                                           &filter_dims,
                                                                           kernel_data,
                                                                           &bias_dims,
                                                                           bias_data,
                                                                           &output_dims,
                                                                           output);

    // Clean up contexts if they were allocated
    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, k7_1d_valid_16ch_32h_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void edge_equal_k_1d_valid_8ch_3h_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[EDGE_EQUAL_K_1D_VALID_8CH_3H_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = EDGE_EQUAL_K_1D_VALID_8CH_3H_DST_SIZE;
    const int32_t *bias_data = get_bias_address(edge_equal_k_1d_valid_8ch_3h_biases, EDGE_EQUAL_K_1D_VALID_8CH_3H_OUT_CH);
    const int8_t *kernel_data = edge_equal_k_1d_valid_8ch_3h_weights;
    const int8_t *input_data = edge_equal_k_1d_valid_8ch_3h_input_tensor;

    // Set up input dimensions: 1D convolution with width=1
    input_dims.n = EDGE_EQUAL_K_1D_VALID_8CH_3H_INPUT_BATCHES;
    input_dims.w = EDGE_EQUAL_K_1D_VALID_8CH_3H_INPUT_W;
    input_dims.h = EDGE_EQUAL_K_1D_VALID_8CH_3H_INPUT_H;
    input_dims.c = EDGE_EQUAL_K_1D_VALID_8CH_3H_IN_CH;

    // Set up filter dimensions: 1D filter with width=1
    filter_dims.w = EDGE_EQUAL_K_1D_VALID_8CH_3H_FILTER_X;
    filter_dims.h = EDGE_EQUAL_K_1D_VALID_8CH_3H_FILTER_Y;
    filter_dims.c = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUT_CH;

    // Set up output dimensions
    output_dims.w = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUTPUT_W;
    output_dims.h = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUTPUT_H;
    output_dims.c = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = EDGE_EQUAL_K_1D_VALID_8CH_3H_PAD_X;
    dw_conv_params.padding.h = EDGE_EQUAL_K_1D_VALID_8CH_3H_PAD_Y;
    dw_conv_params.stride.w = EDGE_EQUAL_K_1D_VALID_8CH_3H_STRIDE_X;
    dw_conv_params.stride.h = EDGE_EQUAL_K_1D_VALID_8CH_3H_STRIDE_Y;
    dw_conv_params.dilation.w = EDGE_EQUAL_K_1D_VALID_8CH_3H_DILATION_X;
    dw_conv_params.dilation.h = EDGE_EQUAL_K_1D_VALID_8CH_3H_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = EDGE_EQUAL_K_1D_VALID_8CH_3H_INPUT_OFFSET;
    dw_conv_params.output_offset = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUTPUT_OFFSET;
    dw_conv_params.activation.min = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = EDGE_EQUAL_K_1D_VALID_8CH_3H_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)edge_equal_k_1d_valid_8ch_3h_output_mult;
    quant_params.shift = (int32_t *)edge_equal_k_1d_valid_8ch_3h_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    // Test the 1D depthwise convolution wrapper function
    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                                           &weights_sum_ctx,
                                                                           &dw_conv_params,
                                                                           &quant_params,
                                                                           &input_dims,
                                                                           input_data,
                                                                           &filter_dims,
                                                                           kernel_data,
                                                                           &bias_dims,
                                                                           bias_data,
                                                                           &output_dims,
                                                                           output);

    // Clean up contexts if they were allocated
    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, edge_equal_k_1d_valid_8ch_3h_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void small_k3_1d_valid_12ch_16h_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[SMALL_K3_1D_VALID_12CH_16H_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = SMALL_K3_1D_VALID_12CH_16H_DST_SIZE;
    const int32_t *bias_data = get_bias_address(small_k3_1d_valid_12ch_16h_biases, SMALL_K3_1D_VALID_12CH_16H_OUT_CH);
    const int8_t *kernel_data = small_k3_1d_valid_12ch_16h_weights;
    const int8_t *input_data = small_k3_1d_valid_12ch_16h_input_tensor;

    // Set up input dimensions: 1D convolution with width=1
    input_dims.n = SMALL_K3_1D_VALID_12CH_16H_INPUT_BATCHES;
    input_dims.w = SMALL_K3_1D_VALID_12CH_16H_INPUT_W;
    input_dims.h = SMALL_K3_1D_VALID_12CH_16H_INPUT_H;
    input_dims.c = SMALL_K3_1D_VALID_12CH_16H_IN_CH;

    // Set up filter dimensions: 1D filter with width=1
    filter_dims.w = SMALL_K3_1D_VALID_12CH_16H_FILTER_X;
    filter_dims.h = SMALL_K3_1D_VALID_12CH_16H_FILTER_Y;
    filter_dims.c = SMALL_K3_1D_VALID_12CH_16H_OUT_CH;

    // Set up output dimensions
    output_dims.w = SMALL_K3_1D_VALID_12CH_16H_OUTPUT_W;
    output_dims.h = SMALL_K3_1D_VALID_12CH_16H_OUTPUT_H;
    output_dims.c = SMALL_K3_1D_VALID_12CH_16H_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = SMALL_K3_1D_VALID_12CH_16H_PAD_X;
    dw_conv_params.padding.h = SMALL_K3_1D_VALID_12CH_16H_PAD_Y;
    dw_conv_params.stride.w = SMALL_K3_1D_VALID_12CH_16H_STRIDE_X;
    dw_conv_params.stride.h = SMALL_K3_1D_VALID_12CH_16H_STRIDE_Y;
    dw_conv_params.dilation.w = SMALL_K3_1D_VALID_12CH_16H_DILATION_X;
    dw_conv_params.dilation.h = SMALL_K3_1D_VALID_12CH_16H_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = SMALL_K3_1D_VALID_12CH_16H_INPUT_OFFSET;
    dw_conv_params.output_offset = SMALL_K3_1D_VALID_12CH_16H_OUTPUT_OFFSET;
    dw_conv_params.activation.min = SMALL_K3_1D_VALID_12CH_16H_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = SMALL_K3_1D_VALID_12CH_16H_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)small_k3_1d_valid_12ch_16h_output_mult;
    quant_params.shift = (int32_t *)small_k3_1d_valid_12ch_16h_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    // Test the 1D depthwise convolution wrapper function
    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                                           &weights_sum_ctx,
                                                                           &dw_conv_params,
                                                                           &quant_params,
                                                                           &input_dims,
                                                                           input_data,
                                                                           &filter_dims,
                                                                           kernel_data,
                                                                           &bias_dims,
                                                                           bias_data,
                                                                           &output_dims,
                                                                           output);

    // Clean up contexts if they were allocated
    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, small_k3_1d_valid_12ch_16h_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void medium_k5_1d_valid_20ch_80h_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[MEDIUM_K5_1D_VALID_20CH_80H_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = MEDIUM_K5_1D_VALID_20CH_80H_DST_SIZE;
    const int32_t *bias_data = get_bias_address(medium_k5_1d_valid_20ch_80h_biases, MEDIUM_K5_1D_VALID_20CH_80H_OUT_CH);
    const int8_t *kernel_data = medium_k5_1d_valid_20ch_80h_weights;
    const int8_t *input_data = medium_k5_1d_valid_20ch_80h_input_tensor;

    // Set up input dimensions: 1D convolution with width=1
    input_dims.n = MEDIUM_K5_1D_VALID_20CH_80H_INPUT_BATCHES;
    input_dims.w = MEDIUM_K5_1D_VALID_20CH_80H_INPUT_W;
    input_dims.h = MEDIUM_K5_1D_VALID_20CH_80H_INPUT_H;
    input_dims.c = MEDIUM_K5_1D_VALID_20CH_80H_IN_CH;

    // Set up filter dimensions: 1D filter with width=1
    filter_dims.w = MEDIUM_K5_1D_VALID_20CH_80H_FILTER_X;
    filter_dims.h = MEDIUM_K5_1D_VALID_20CH_80H_FILTER_Y;
    filter_dims.c = MEDIUM_K5_1D_VALID_20CH_80H_OUT_CH;

    // Set up output dimensions
    output_dims.w = MEDIUM_K5_1D_VALID_20CH_80H_OUTPUT_W;
    output_dims.h = MEDIUM_K5_1D_VALID_20CH_80H_OUTPUT_H;
    output_dims.c = MEDIUM_K5_1D_VALID_20CH_80H_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = MEDIUM_K5_1D_VALID_20CH_80H_PAD_X;
    dw_conv_params.padding.h = MEDIUM_K5_1D_VALID_20CH_80H_PAD_Y;
    dw_conv_params.stride.w = MEDIUM_K5_1D_VALID_20CH_80H_STRIDE_X;
    dw_conv_params.stride.h = MEDIUM_K5_1D_VALID_20CH_80H_STRIDE_Y;
    dw_conv_params.dilation.w = MEDIUM_K5_1D_VALID_20CH_80H_DILATION_X;
    dw_conv_params.dilation.h = MEDIUM_K5_1D_VALID_20CH_80H_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = MEDIUM_K5_1D_VALID_20CH_80H_INPUT_OFFSET;
    dw_conv_params.output_offset = MEDIUM_K5_1D_VALID_20CH_80H_OUTPUT_OFFSET;
    dw_conv_params.activation.min = MEDIUM_K5_1D_VALID_20CH_80H_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = MEDIUM_K5_1D_VALID_20CH_80H_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)medium_k5_1d_valid_20ch_80h_output_mult;
    quant_params.shift = (int32_t *)medium_k5_1d_valid_20ch_80h_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    // Test the 1D depthwise convolution wrapper function
    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                                           &weights_sum_ctx,
                                                                           &dw_conv_params,
                                                                           &quant_params,
                                                                           &input_dims,
                                                                           input_data,
                                                                           &filter_dims,
                                                                           kernel_data,
                                                                           &bias_dims,
                                                                           bias_data,
                                                                           &output_dims,
                                                                           output);

    // Clean up contexts if they were allocated
    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, medium_k5_1d_valid_20ch_80h_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void large_k9_1d_valid_28ch_200h_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[LARGE_K9_1D_VALID_28CH_200H_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = LARGE_K9_1D_VALID_28CH_200H_DST_SIZE;
    const int32_t *bias_data = get_bias_address(large_k9_1d_valid_28ch_200h_biases, LARGE_K9_1D_VALID_28CH_200H_OUT_CH);
    const int8_t *kernel_data = large_k9_1d_valid_28ch_200h_weights;
    const int8_t *input_data = large_k9_1d_valid_28ch_200h_input_tensor;

    // Set up input dimensions: 1D convolution with width=1
    input_dims.n = LARGE_K9_1D_VALID_28CH_200H_INPUT_BATCHES;
    input_dims.w = LARGE_K9_1D_VALID_28CH_200H_INPUT_W;
    input_dims.h = LARGE_K9_1D_VALID_28CH_200H_INPUT_H;
    input_dims.c = LARGE_K9_1D_VALID_28CH_200H_IN_CH;

    // Set up filter dimensions: 1D filter with width=1
    filter_dims.w = LARGE_K9_1D_VALID_28CH_200H_FILTER_X;
    filter_dims.h = LARGE_K9_1D_VALID_28CH_200H_FILTER_Y;
    filter_dims.c = LARGE_K9_1D_VALID_28CH_200H_OUT_CH;

    // Set up output dimensions
    output_dims.w = LARGE_K9_1D_VALID_28CH_200H_OUTPUT_W;
    output_dims.h = LARGE_K9_1D_VALID_28CH_200H_OUTPUT_H;
    output_dims.c = LARGE_K9_1D_VALID_28CH_200H_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = LARGE_K9_1D_VALID_28CH_200H_PAD_X;
    dw_conv_params.padding.h = LARGE_K9_1D_VALID_28CH_200H_PAD_Y;
    dw_conv_params.stride.w = LARGE_K9_1D_VALID_28CH_200H_STRIDE_X;
    dw_conv_params.stride.h = LARGE_K9_1D_VALID_28CH_200H_STRIDE_Y;
    dw_conv_params.dilation.w = LARGE_K9_1D_VALID_28CH_200H_DILATION_X;
    dw_conv_params.dilation.h = LARGE_K9_1D_VALID_28CH_200H_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = LARGE_K9_1D_VALID_28CH_200H_INPUT_OFFSET;
    dw_conv_params.output_offset = LARGE_K9_1D_VALID_28CH_200H_OUTPUT_OFFSET;
    dw_conv_params.activation.min = LARGE_K9_1D_VALID_28CH_200H_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = LARGE_K9_1D_VALID_28CH_200H_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)large_k9_1d_valid_28ch_200h_output_mult;
    quant_params.shift = (int32_t *)large_k9_1d_valid_28ch_200h_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    // Test the 1D depthwise convolution wrapper function
    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                                           &weights_sum_ctx,
                                                                           &dw_conv_params,
                                                                           &quant_params,
                                                                           &input_dims,
                                                                           input_data,
                                                                           &filter_dims,
                                                                           kernel_data,
                                                                           &bias_dims,
                                                                           bias_data,
                                                                           &output_dims,
                                                                           output);

    // Clean up contexts if they were allocated
    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, large_k9_1d_valid_28ch_200h_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void edge_equal_k_1d_valid_128ch_15w_arm_depthwise_conv_s8_opt_1d_valid_wrapper(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[EDGE_EQUAL_K_1D_VALID_128CH_15W_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(edge_equal_k_1d_valid_128ch_15w_biases, EDGE_EQUAL_K_1D_VALID_128CH_15W_OUT_CH);
    const int8_t *kernel_data = edge_equal_k_1d_valid_128ch_15w_weights;
    const int8_t *input_data = edge_equal_k_1d_valid_128ch_15w_input_tensor;

    // Set up input dimensions: 1D convolution with height=1
    input_dims.n = EDGE_EQUAL_K_1D_VALID_128CH_15W_INPUT_BATCHES;
    input_dims.w = EDGE_EQUAL_K_1D_VALID_128CH_15W_INPUT_W;
    input_dims.h = EDGE_EQUAL_K_1D_VALID_128CH_15W_INPUT_H;
    input_dims.c = EDGE_EQUAL_K_1D_VALID_128CH_15W_IN_CH;

    // Set up filter dimensions: 1D filter with height=1
    filter_dims.w = EDGE_EQUAL_K_1D_VALID_128CH_15W_FILTER_X;
    filter_dims.h = EDGE_EQUAL_K_1D_VALID_128CH_15W_FILTER_Y;
    filter_dims.c = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUT_CH;

    // Set up output dimensions
    output_dims.w = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUTPUT_W;
    output_dims.h = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUTPUT_H;
    output_dims.c = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUT_CH;

    // Set up convolution parameters
    dw_conv_params.padding.w = EDGE_EQUAL_K_1D_VALID_128CH_15W_PAD_X;
    dw_conv_params.padding.h = EDGE_EQUAL_K_1D_VALID_128CH_15W_PAD_Y;
    dw_conv_params.stride.w = EDGE_EQUAL_K_1D_VALID_128CH_15W_STRIDE_X;
    dw_conv_params.stride.h = EDGE_EQUAL_K_1D_VALID_128CH_15W_STRIDE_Y;
    dw_conv_params.dilation.w = EDGE_EQUAL_K_1D_VALID_128CH_15W_DILATION_X;
    dw_conv_params.dilation.h = EDGE_EQUAL_K_1D_VALID_128CH_15W_DILATION_Y;
    dw_conv_params.ch_mult = 1; // Channel multiplier = 1 for depthwise convolution

    dw_conv_params.input_offset = EDGE_EQUAL_K_1D_VALID_128CH_15W_INPUT_OFFSET;
    dw_conv_params.output_offset = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUTPUT_OFFSET;
    dw_conv_params.activation.min = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = EDGE_EQUAL_K_1D_VALID_128CH_15W_OUT_ACTIVATION_MAX;

    // Set up per-channel quantization parameters
    quant_params.multiplier = (int32_t *)edge_equal_k_1d_valid_128ch_15w_output_mult;
    quant_params.shift = (int32_t *)edge_equal_k_1d_valid_128ch_15w_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_1d_wrapper_s8(&ctx,
                                                           &weights_sum_ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           kernel_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, edge_equal_k_1d_valid_128ch_15w_output_ref, EDGE_EQUAL_K_1D_VALID_128CH_15W_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t*)weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
                                           &dw_conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input_data,
                                           &filter_dims,
                                           kernel_data,
                                           &bias_dims,
                                           bias_data,
                                           &output_dims,
                                           output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, edge_equal_k_1d_valid_128ch_15w_output_ref, EDGE_EQUAL_K_1D_VALID_128CH_15W_DST_SIZE));
}