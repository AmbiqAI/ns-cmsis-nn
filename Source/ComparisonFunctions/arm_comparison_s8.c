/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_comparison_s8
 * Description:  Elementwise comparisons with broadcasting support for s8
 *
 * $Date:        3 November 2025
 * $Revision:    V.0.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Comparison
 * @{
 */

typedef struct
{
    int32_t offset;
    int32_t multiplier;
    int32_t shift;
} arm_compare_quant_params;

/** Helper to validate supported comparison operations */
static inline bool arm_comparison_operation_is_valid(arm_nn_compare_operation operation)
{
    return (operation <= ARM_COMPARE_LESS_EQUAL);
}

static inline int32_t
arm_prepare_value_s8(int32_t value, const arm_compare_quant_params *params, const int32_t left_shift)
{
    int32_t res = value + params->offset;
    res = res << left_shift;
    return arm_nn_requantize(res, params->multiplier, params->shift);
}

static inline int32_t
arm_compare_rescaled_s32(const int32_t lhs, const int32_t rhs, const arm_nn_compare_operation operation)
{
    switch (operation)
    {
    case ARM_COMPARE_EQUAL:
        return lhs == rhs;
    case ARM_COMPARE_NOT_EQUAL:
        return lhs != rhs;
    case ARM_COMPARE_GREATER:
        return lhs > rhs;
    case ARM_COMPARE_GREATER_EQUAL:
        return lhs >= rhs;
    case ARM_COMPARE_LESS:
        return lhs < rhs;
    case ARM_COMPARE_LESS_EQUAL:
        return lhs <= rhs;
    default:
        return 0;
    }
}

#if defined(ARM_MATH_MVEI)
static inline int32x4_t
arm_prepare_vector_s8(int32x4_t value, const arm_compare_quant_params *params, const int32_t left_shift)
{
    value = vaddq_n_s32(value, params->offset);
    value = vshlq_r_s32(value, left_shift);
    return arm_requantize_mve(value, params->multiplier, params->shift);
}

static inline mve_pred16_t
arm_compare_mask_vec_s32(const int32x4_t lhs, const int32x4_t rhs, const arm_nn_compare_operation operation)
{
    switch (operation)
    {
    case ARM_COMPARE_EQUAL:
        return vcmpeqq_s32(lhs, rhs);
    case ARM_COMPARE_NOT_EQUAL:
        return vcmpneq_s32(lhs, rhs);
    case ARM_COMPARE_GREATER:
        return vcmpgtq_s32(lhs, rhs);
    case ARM_COMPARE_GREATER_EQUAL:
        return vcmpgeq_s32(lhs, rhs);
    case ARM_COMPARE_LESS:
        return vcmpltq_s32(lhs, rhs);
    case ARM_COMPARE_LESS_EQUAL:
        return vcmpleq_s32(lhs, rhs);
    default:
        return vcmpneq_s32(lhs, lhs);
    }
}
#endif

static arm_cmsis_nn_status arm_compare_no_broadcast_s8(const int8_t *input_1,
                                                       const int8_t *input_2,
                                                       bool *output,
                                                       int32_t block_size,
                                                       const arm_compare_quant_params *input_1_params,
                                                       const arm_compare_quant_params *input_2_params,
                                                       const int32_t left_shift,
                                                       const arm_nn_compare_operation operation)
{
#if defined(ARM_MATH_MVEI)
    int32_t count = block_size;
    const int8_t *lhs_ptr = input_1;
    const int8_t *rhs_ptr = input_2;
    bool *out_ptr = output;
    const int32x4_t zeros = vdupq_n_s32(0);
    const int32x4_t ones = vdupq_n_s32(1);

    while (count > 0)
    {
        mve_pred16_t p = vctp32q((uint32_t)count);
        int32x4_t lhs = vldrbq_z_s32(lhs_ptr, p);
        int32x4_t rhs = vldrbq_z_s32(rhs_ptr, p);

        lhs = arm_prepare_vector_s8(lhs, input_1_params, left_shift);
        rhs = arm_prepare_vector_s8(rhs, input_2_params, left_shift);

        mve_pred16_t cmp = arm_compare_mask_vec_s32(lhs, rhs, operation);
        int32x4_t bool_vec = vpselq_s32(ones, zeros, cmp);
        vstrbq_p_s32((int8_t *)out_ptr, bool_vec, p);

        lhs_ptr += 4;
        rhs_ptr += 4;
        out_ptr += 4;
        count -= 4;
    }

    return ARM_CMSIS_NN_SUCCESS;
#else
    const int8_t *lhs_ptr = input_1;
    const int8_t *rhs_ptr = input_2;
    bool *out_ptr = output;

    #if defined(ARM_MATH_DSP)
    int32_t loop_count = block_size >> 2;
    int32_t tail = block_size & 0x3;

    while (loop_count-- > 0)
    {
        int32_t lhs_packed = arm_nn_read_s8x4_ia(&lhs_ptr);
        int32_t rhs_packed = arm_nn_read_s8x4_ia(&rhs_ptr);

        int32_t lhs0 = arm_prepare_value_s8((int8_t)lhs_packed, input_1_params, left_shift);
        int32_t rhs0 = arm_prepare_value_s8((int8_t)rhs_packed, input_2_params, left_shift);

        int32_t lhs1 = arm_prepare_value_s8((int8_t)(lhs_packed >> 8), input_1_params, left_shift);
        int32_t rhs1 = arm_prepare_value_s8((int8_t)(rhs_packed >> 8), input_2_params, left_shift);

        int32_t lhs2 = arm_prepare_value_s8((int8_t)(lhs_packed >> 16), input_1_params, left_shift);
        int32_t rhs2 = arm_prepare_value_s8((int8_t)(rhs_packed >> 16), input_2_params, left_shift);

        int32_t lhs3 = arm_prepare_value_s8((int8_t)(lhs_packed >> 24), input_1_params, left_shift);
        int32_t rhs3 = arm_prepare_value_s8((int8_t)(rhs_packed >> 24), input_2_params, left_shift);

        bool r0 = arm_compare_rescaled_s32(lhs0, rhs0, operation);
        bool r1 = arm_compare_rescaled_s32(lhs1, rhs1, operation);
        bool r2 = arm_compare_rescaled_s32(lhs2, rhs2, operation);
        bool r3 = arm_compare_rescaled_s32(lhs3, rhs3, operation);

        *out_ptr++ = r0;
        *out_ptr++ = r1;
        *out_ptr++ = r2;
        *out_ptr++ = r3;
    }
    #else
    int32_t tail = block_size;
    #endif

    while (tail-- > 0)
    {
        int32_t lhs = arm_prepare_value_s8(*lhs_ptr++, input_1_params, left_shift);
        int32_t rhs = arm_prepare_value_s8(*rhs_ptr++, input_2_params, left_shift);
        *out_ptr++ = arm_compare_rescaled_s32(lhs, rhs, operation);
    }

    return ARM_CMSIS_NN_SUCCESS;
#endif
}

static arm_cmsis_nn_status arm_compare_scalar_broadcast_s8(int8_t scalar_value,
                                                           const arm_compare_quant_params *scalar_params,
                                                           const int8_t *tensor,
                                                           const arm_compare_quant_params *tensor_params,
                                                           bool *output,
                                                           int32_t block_size,
                                                           int32_t left_shift,
                                                           bool scalar_is_lhs,
                                                           arm_nn_compare_operation operation)
{
    const int32_t scalar_rescaled = arm_prepare_value_s8(scalar_value, scalar_params, left_shift);

#if defined(ARM_MATH_MVEI)
    int32_t count = block_size;
    const int8_t *tensor_ptr = tensor;
    bool *out_ptr = output;
    const int32x4_t scalar_vec = vdupq_n_s32(scalar_rescaled);
    const int32x4_t zeros = vdupq_n_s32(0);
    const int32x4_t ones = vdupq_n_s32(1);

    while (count > 0)
    {
        mve_pred16_t p = vctp32q((uint32_t)count);
        int32x4_t tensor_vec = vldrbq_z_s32(tensor_ptr, p);
        tensor_vec = arm_prepare_vector_s8(tensor_vec, tensor_params, left_shift);

        mve_pred16_t cmp = scalar_is_lhs ? arm_compare_mask_vec_s32(scalar_vec, tensor_vec, operation)
                                         : arm_compare_mask_vec_s32(tensor_vec, scalar_vec, operation);
        int32x4_t bool_vec = vpselq_s32(ones, zeros, cmp);
        vstrbq_p_s32((int8_t *)out_ptr, bool_vec, p);

        tensor_ptr += 4;
        out_ptr += 4;
        count -= 4;
    }

    return ARM_CMSIS_NN_SUCCESS;
#else
    const int8_t *tensor_ptr = tensor;
    bool *out_ptr = output;

    #if defined(ARM_MATH_DSP)
    int32_t loop_count = block_size >> 2;
    int32_t tail = block_size & 0x3;

    while (loop_count-- > 0)
    {
        int32_t tensor_packed = arm_nn_read_s8x4_ia(&tensor_ptr);

        int32_t tensor0 = arm_prepare_value_s8((int8_t)tensor_packed, tensor_params, left_shift);
        int32_t tensor1 = arm_prepare_value_s8((int8_t)(tensor_packed >> 8), tensor_params, left_shift);
        int32_t tensor2 = arm_prepare_value_s8((int8_t)(tensor_packed >> 16), tensor_params, left_shift);
        int32_t tensor3 = arm_prepare_value_s8((int8_t)(tensor_packed >> 24), tensor_params, left_shift);

        bool r0 = scalar_is_lhs ? arm_compare_rescaled_s32(scalar_rescaled, tensor0, operation)
                                : arm_compare_rescaled_s32(tensor0, scalar_rescaled, operation);
        bool r1 = scalar_is_lhs ? arm_compare_rescaled_s32(scalar_rescaled, tensor1, operation)
                                : arm_compare_rescaled_s32(tensor1, scalar_rescaled, operation);
        bool r2 = scalar_is_lhs ? arm_compare_rescaled_s32(scalar_rescaled, tensor2, operation)
                                : arm_compare_rescaled_s32(tensor2, scalar_rescaled, operation);
        bool r3 = scalar_is_lhs ? arm_compare_rescaled_s32(scalar_rescaled, tensor3, operation)
                                : arm_compare_rescaled_s32(tensor3, scalar_rescaled, operation);

        *out_ptr++ = r0;
        *out_ptr++ = r1;
        *out_ptr++ = r2;
        *out_ptr++ = r3;
    }
    #else
    int32_t tail = block_size;
    #endif

    while (tail-- > 0)
    {
        int32_t tensor_val = arm_prepare_value_s8(*tensor_ptr++, tensor_params, left_shift);
        *out_ptr++ = scalar_is_lhs ? arm_compare_rescaled_s32(scalar_rescaled, tensor_val, operation)
                                   : arm_compare_rescaled_s32(tensor_val, scalar_rescaled, operation);
    }

    return ARM_CMSIS_NN_SUCCESS;
#endif
}

arm_cmsis_nn_status arm_comparison_s8(const cmsis_nn_context *ctx,
                                      const int8_t *input_1_data,
                                      const cmsis_nn_dims *input_1_dims,
                                      const int8_t *input_2_data,
                                      const cmsis_nn_dims *input_2_dims,
                                      bool *output_data,
                                      const cmsis_nn_dims *output_dims,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift,
                                      const arm_nn_compare_operation operation)
{
    (void)ctx;

    if (!arm_comparison_operation_is_valid(operation))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const arm_compare_quant_params input_1_params = {input_1_offset, input_1_mult, input_1_shift};
    const arm_compare_quant_params input_2_params = {input_2_offset, input_2_mult, input_2_shift};

    const int32_t output_batch = output_dims->n;
    const int32_t output_height = output_dims->h;
    const int32_t output_width = output_dims->w;

    const int32_t input_1_batch = input_1_dims->n;
    const int32_t input_1_height = input_1_dims->h;
    const int32_t input_1_width = input_1_dims->w;
    const int32_t input_1_channels = input_1_dims->c;

    const int32_t input_2_batch = input_2_dims->n;
    const int32_t input_2_height = input_2_dims->h;
    const int32_t input_2_width = input_2_dims->w;
    const int32_t input_2_channels = input_2_dims->c;

    int32_t flat_size_1 = input_1_batch * input_1_height * input_1_width * input_1_channels;
    int32_t flat_size_2 = input_2_batch * input_2_height * input_2_width * input_2_channels;

    if (!arm_check_broadcast_required(input_1_dims, input_2_dims))
    {
        return arm_compare_no_broadcast_s8(input_1_data,
                                           input_2_data,
                                           output_data,
                                           flat_size_1,
                                           &input_1_params,
                                           &input_2_params,
                                           left_shift,
                                           operation);
    }

    if (flat_size_1 == 1)
    {
        return arm_compare_scalar_broadcast_s8(*input_1_data,
                                               &input_1_params,
                                               input_2_data,
                                               &input_2_params,
                                               output_data,
                                               flat_size_2,
                                               left_shift,
                                               true,
                                               operation);
    }

    if (flat_size_2 == 1)
    {
        return arm_compare_scalar_broadcast_s8(*input_2_data,
                                               &input_2_params,
                                               input_1_data,
                                               &input_1_params,
                                               output_data,
                                               flat_size_1,
                                               left_shift,
                                               false,
                                               operation);
    }

    int32_t width_1_diff = (input_1_width >= input_2_width) ? 0 : input_1_channels;
    int32_t width_2_diff = (input_2_width >= input_1_width) ? 0 : input_2_channels;

    int32_t height_1_diff =
        (input_1_height >= input_2_height) ? width_1_diff : -input_1_width * (input_1_channels - width_1_diff);
    int32_t height_2_diff =
        (input_2_height >= input_1_height) ? width_2_diff : -input_2_width * (input_2_channels - width_2_diff);

    int32_t batch_1_diff = (input_1_batch >= input_2_batch) ? input_1_channels * input_1_width * input_1_height : 0;
    int32_t batch_2_diff = (input_2_batch >= input_1_batch) ? input_2_channels * input_2_width * input_2_height : 0;

    for (int32_t i_out_batch = 0; i_out_batch < output_batch; i_out_batch++)
    {
        const int8_t *input_1_ptr = input_1_data;
        const int8_t *input_2_ptr = input_2_data;

        flat_size_1 = input_1_height * input_1_width * input_1_channels;
        flat_size_2 = input_2_height * input_2_width * input_2_channels;

        if ((input_1_height == input_2_height) && (input_1_width == input_2_width) &&
            (input_1_channels == input_2_channels))
        {
            arm_compare_no_broadcast_s8(input_1_ptr,
                                        input_2_ptr,
                                        output_data,
                                        flat_size_1,
                                        &input_1_params,
                                        &input_2_params,
                                        left_shift,
                                        operation);

            input_1_ptr += flat_size_1;
            input_2_ptr += flat_size_1;
            output_data += flat_size_1;
        }
        else if (flat_size_1 == 1)
        {
            arm_compare_scalar_broadcast_s8(*input_1_ptr,
                                            &input_1_params,
                                            input_2_ptr,
                                            &input_2_params,
                                            output_data,
                                            flat_size_2,
                                            left_shift,
                                            true,
                                            operation);

            ++input_1_ptr;
            input_2_ptr += flat_size_2;
            output_data += flat_size_2;
        }
        else if (flat_size_2 == 1)
        {
            arm_compare_scalar_broadcast_s8(*input_2_ptr,
                                            &input_2_params,
                                            input_1_ptr,
                                            &input_1_params,
                                            output_data,
                                            flat_size_1,
                                            left_shift,
                                            false,
                                            operation);

            ++input_2_ptr;
            input_1_ptr += flat_size_1;
            output_data += flat_size_1;
        }
        else
        {
            flat_size_1 = input_1_width * input_1_channels;
            flat_size_2 = input_2_width * input_2_channels;
            for (int32_t i_out_height = 0; i_out_height < output_height; i_out_height++)
            {
                if ((input_1_width == input_2_width) && (input_1_channels == input_2_channels))
                {
                    arm_compare_no_broadcast_s8(input_1_ptr,
                                                input_2_ptr,
                                                output_data,
                                                flat_size_1,
                                                &input_1_params,
                                                &input_2_params,
                                                left_shift,
                                                operation);

                    input_1_ptr += flat_size_1;
                    input_2_ptr += flat_size_1;
                    output_data += flat_size_1;
                }
                else if (flat_size_1 == 1)
                {
                    arm_compare_scalar_broadcast_s8(*input_1_ptr,
                                                    &input_1_params,
                                                    input_2_ptr,
                                                    &input_2_params,
                                                    output_data,
                                                    flat_size_2,
                                                    left_shift,
                                                    true,
                                                    operation);

                    input_2_ptr += flat_size_2;
                    output_data += flat_size_2;
                }
                else if (flat_size_2 == 1)
                {
                    arm_compare_scalar_broadcast_s8(*input_2_ptr,
                                                    &input_2_params,
                                                    input_1_ptr,
                                                    &input_1_params,
                                                    output_data,
                                                    flat_size_1,
                                                    left_shift,
                                                    false,
                                                    operation);

                    input_1_ptr += flat_size_1;
                    output_data += flat_size_1;
                }
                else
                {
                    for (int32_t i_out_width = 0; i_out_width < output_width; i_out_width++)
                    {
                        if (input_1_channels == input_2_channels)
                        {
                            arm_compare_no_broadcast_s8(input_1_ptr,
                                                        input_2_ptr,
                                                        output_data,
                                                        input_1_channels,
                                                        &input_1_params,
                                                        &input_2_params,
                                                        left_shift,
                                                        operation);

                            input_1_ptr += input_1_channels;
                            input_2_ptr += input_1_channels;
                            output_data += input_1_channels;
                        }
                        else if (input_1_channels == 1)
                        {
                            arm_compare_scalar_broadcast_s8(*input_1_ptr,
                                                            &input_1_params,
                                                            input_2_ptr,
                                                            &input_2_params,
                                                            output_data,
                                                            input_2_channels,
                                                            left_shift,
                                                            true,
                                                            operation);

                            ++input_1_ptr;
                            input_2_ptr += input_2_channels;
                            output_data += input_2_channels;
                        }
                        else if (input_2_channels == 1)
                        {
                            arm_compare_scalar_broadcast_s8(*input_2_ptr,
                                                            &input_2_params,
                                                            input_1_ptr,
                                                            &input_1_params,
                                                            output_data,
                                                            input_1_channels,
                                                            left_shift,
                                                            false,
                                                            operation);

                            input_1_ptr += input_1_channels;
                            ++input_2_ptr;
                            output_data += input_1_channels;
                        }

                        input_1_ptr -= width_1_diff;
                        input_2_ptr -= width_2_diff;
                    }
                }

                input_1_ptr += height_1_diff;
                input_2_ptr += height_2_diff;
            }
        }

        input_1_data += batch_1_diff;
        input_2_data += batch_2_diff;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_equal_s8(const cmsis_nn_context *ctx,
                                 const int8_t *input_1_data,
                                 const cmsis_nn_dims *input_1_dims,
                                 const int8_t *input_2_data,
                                 const cmsis_nn_dims *input_2_dims,
                                 bool *output_data,
                                 const cmsis_nn_dims *output_dims,
                                 const int32_t input_1_offset,
                                 const int32_t input_1_mult,
                                 const int32_t input_1_shift,
                                 const int32_t input_2_offset,
                                 const int32_t input_2_mult,
                                 const int32_t input_2_shift,
                                 const int32_t left_shift)
{
    return arm_comparison_s8(ctx,
                             input_1_data,
                             input_1_dims,
                             input_2_data,
                             input_2_dims,
                             output_data,
                             output_dims,
                             input_1_offset,
                             input_1_mult,
                             input_1_shift,
                             input_2_offset,
                             input_2_mult,
                             input_2_shift,
                             left_shift,
                             ARM_COMPARE_EQUAL);
}

arm_cmsis_nn_status arm_not_equal_s8(const cmsis_nn_context *ctx,
                                     const int8_t *input_1_data,
                                     const cmsis_nn_dims *input_1_dims,
                                     const int8_t *input_2_data,
                                     const cmsis_nn_dims *input_2_dims,
                                     bool *output_data,
                                     const cmsis_nn_dims *output_dims,
                                     const int32_t input_1_offset,
                                     const int32_t input_1_mult,
                                     const int32_t input_1_shift,
                                     const int32_t input_2_offset,
                                     const int32_t input_2_mult,
                                     const int32_t input_2_shift,
                                     const int32_t left_shift)
{
    return arm_comparison_s8(ctx,
                             input_1_data,
                             input_1_dims,
                             input_2_data,
                             input_2_dims,
                             output_data,
                             output_dims,
                             input_1_offset,
                             input_1_mult,
                             input_1_shift,
                             input_2_offset,
                             input_2_mult,
                             input_2_shift,
                             left_shift,
                             ARM_COMPARE_NOT_EQUAL);
}

arm_cmsis_nn_status arm_greater_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   bool *output_data,
                                   const cmsis_nn_dims *output_dims,
                                   const int32_t input_1_offset,
                                   const int32_t input_1_mult,
                                   const int32_t input_1_shift,
                                   const int32_t input_2_offset,
                                   const int32_t input_2_mult,
                                   const int32_t input_2_shift,
                                   const int32_t left_shift)
{
    return arm_comparison_s8(ctx,
                             input_1_data,
                             input_1_dims,
                             input_2_data,
                             input_2_dims,
                             output_data,
                             output_dims,
                             input_1_offset,
                             input_1_mult,
                             input_1_shift,
                             input_2_offset,
                             input_2_mult,
                             input_2_shift,
                             left_shift,
                             ARM_COMPARE_GREATER);
}

arm_cmsis_nn_status arm_greater_equal_s8(const cmsis_nn_context *ctx,
                                         const int8_t *input_1_data,
                                         const cmsis_nn_dims *input_1_dims,
                                         const int8_t *input_2_data,
                                         const cmsis_nn_dims *input_2_dims,
                                         bool *output_data,
                                         const cmsis_nn_dims *output_dims,
                                         const int32_t input_1_offset,
                                         const int32_t input_1_mult,
                                         const int32_t input_1_shift,
                                         const int32_t input_2_offset,
                                         const int32_t input_2_mult,
                                         const int32_t input_2_shift,
                                         const int32_t left_shift)
{
    return arm_comparison_s8(ctx,
                             input_1_data,
                             input_1_dims,
                             input_2_data,
                             input_2_dims,
                             output_data,
                             output_dims,
                             input_1_offset,
                             input_1_mult,
                             input_1_shift,
                             input_2_offset,
                             input_2_mult,
                             input_2_shift,
                             left_shift,
                             ARM_COMPARE_GREATER_EQUAL);
}

arm_cmsis_nn_status arm_less_s8(const cmsis_nn_context *ctx,
                                const int8_t *input_1_data,
                                const cmsis_nn_dims *input_1_dims,
                                const int8_t *input_2_data,
                                const cmsis_nn_dims *input_2_dims,
                                bool *output_data,
                                const cmsis_nn_dims *output_dims,
                                const int32_t input_1_offset,
                                const int32_t input_1_mult,
                                const int32_t input_1_shift,
                                const int32_t input_2_offset,
                                const int32_t input_2_mult,
                                const int32_t input_2_shift,
                                const int32_t left_shift)
{
    return arm_comparison_s8(ctx,
                             input_1_data,
                             input_1_dims,
                             input_2_data,
                             input_2_dims,
                             output_data,
                             output_dims,
                             input_1_offset,
                             input_1_mult,
                             input_1_shift,
                             input_2_offset,
                             input_2_mult,
                             input_2_shift,
                             left_shift,
                             ARM_COMPARE_LESS);
}

arm_cmsis_nn_status arm_less_equal_s8(const cmsis_nn_context *ctx,
                                      const int8_t *input_1_data,
                                      const cmsis_nn_dims *input_1_dims,
                                      const int8_t *input_2_data,
                                      const cmsis_nn_dims *input_2_dims,
                                      bool *output_data,
                                      const cmsis_nn_dims *output_dims,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift)
{
    return arm_comparison_s8(ctx,
                             input_1_data,
                             input_1_dims,
                             input_2_data,
                             input_2_dims,
                             output_data,
                             output_dims,
                             input_1_offset,
                             input_1_mult,
                             input_1_shift,
                             input_2_offset,
                             input_2_mult,
                             input_2_shift,
                             left_shift,
                             ARM_COMPARE_LESS_EQUAL);
}

/**
 * @} end of Doxygen group
 */
