/*
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_svdf_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for svdf s8 layer function.
 *
 * $Date:        5 September 2023
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup SVDF
 */

/**
 * @addtogroup GetBufferSizeSVDF
 * @{
 */

int32_t arm_svdf_s8_get_buffer_size_dsp(const cmsis_nn_dims *weights_feature_dims)
{
    // This leg needs no buffer, but it is a public entry point that the Python bindings call directly, so it has
    // to reject an out-of-range weights_feature_dims->n with the same -1 the dispatcher and the MVE leg return
    // rather than a 0 the caller would read as "no buffer needed" (issue #349). The bound is the dispatcher's, not
    // this leg's: no buffer is sized here, so the check exists only to keep the three entry points answering alike.
    const int64_t required_bytes = (int64_t)weights_feature_dims->n * (int64_t)sizeof(int32_t);

    if ((weights_feature_dims->n < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return 0;
}

int32_t arm_svdf_s8_get_buffer_size_mve(const cmsis_nn_dims *weights_feature_dims)
{
    // Computed in 64 bits so that a row count large enough to overflow the int32_t byte count cannot wrap past
    // the range check below.
    const int64_t required_bytes = (int64_t)weights_feature_dims->n * (int64_t)sizeof(int32_t);

    if ((weights_feature_dims->n < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return (int32_t)required_bytes;
}

int32_t arm_svdf_s8_get_buffer_size(const cmsis_nn_dims *weights_feature_dims)
{
    // Validated once here, ahead of the dispatch below, so an invalid dim returns -1 on every build target.
    // Both leg variants are also called directly by binding glue and re-check this on their own.
    const int64_t required_bytes = (int64_t)weights_feature_dims->n * (int64_t)sizeof(int32_t);

    if ((weights_feature_dims->n < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

#if defined(ARM_MATH_MVEI)
    return arm_svdf_s8_get_buffer_size_mve(weights_feature_dims);
#else
    return arm_svdf_s8_get_buffer_size_dsp(weights_feature_dims);
#endif
}

/*
 * The input_ctx / output_ctx scratch sizes below are shared by arm_svdf_s8() and arm_svdf_state_s16_s8(). Both
 * kernels stage the same two int32_t accumulator arrays regardless of their state/time-weight width, and both do
 * so on every build target - these buffers are not MVE-gated - so there is deliberately no _dsp / _mve pair here.
 * A single non-dispatching query keeps the answer identical across targets, which is what a caller sizing an
 * arena ahead of time needs.
 */

/*
 * Bytes staged in input_ctx: one int32_t accumulator per (input batch, feature batch). Derived from the kernel,
 * not from a shape table: in both arm_svdf_s8() and arm_svdf_state_s16_s8() the "time weight * state" block walks
 * a single cursor `ptr_a` from the base of input_ctx->buf, storing exactly once per iteration of the nested
 * i_batch (input_dims->n) / i_feature_batch (weights_feature_dims->n) loops and never rewinding. The highest
 * element written is therefore index input_dims->n * weights_feature_dims->n - 1.
 */
static int64_t arm_svdf_input_ctx_bytes(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *weights_feature_dims)
{
    if ((input_dims == NULL) || (weights_feature_dims == NULL))
    {
        return -1;
    }

    // Folded one factor at a time so the accumulator stays bounded; see arm_nn_size_mul().
    int64_t required_bytes = arm_nn_size_mul(1, input_dims->n);
    required_bytes = arm_nn_size_mul(required_bytes, weights_feature_dims->n);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int32_t));

    return required_bytes;
}

/*
 * Bytes staged in output_ctx: one int32_t accumulator per (input batch, output unit). Derived from the kernel:
 * every branch of the bias/no-bias block writes through `buffer_b + i_batch * unit_count` for indices
 * [0, unit_count), where unit_count = weights_feature_dims->n / svdf_params->rank. The one branch that indexes by
 * feature_batches instead is guarded by `unit_count == feature_batches`, so it writes the same extent. The MVE
 * requantization tail reads buffer_b through vldrwq_z_s32() under a vctp32q() predicate, so it does not touch the
 * lanes past num_elements and adds no round-up to the requirement.
 *
 * The division truncates, exactly as the kernel's `feature_batches / rank` does. Rounding up here would over-report
 * for the (ill-formed) case where rank does not divide weights_feature_dims->n, and reporting the kernel's own
 * arithmetic is what makes this number checkable against it.
 */
static int64_t arm_svdf_output_ctx_bytes(const int32_t rank,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *weights_feature_dims)
{
    if ((input_dims == NULL) || (weights_feature_dims == NULL))
    {
        return -1;
    }

    /* Both integer kernels narrow rank to int16_t before dividing - `const int16_t rank = svdf_params->rank` in
       arm_svdf_s8.c and arm_svdf_state_s16_s8.c - while cmsis_nn_svdf_params::rank is int32_t. A rank that does
       not survive that round trip would make this query disagree with the kernel it is supposed to describe:
       rank 65538 narrows to 2, so for weights_feature_dims->n == 100 the kernel writes 50 units (200 bytes)
       while this formula would report 100 / 65538 == 0. Reject it instead of publishing a number the kernel will
       not honour. rank 65536 narrows to 0, which the same test catches before it can reach the division.

       This validates the sizer against the kernel as it is. Widening the kernels' rank to int32_t is the
       root-cause fix, but it touches perf-sensitive upstream-inherited code and belongs in its own change. */
    if (rank != (int32_t)(int16_t)rank)
    {
        return -1;
    }

    // rank is a divisor here, so it must be validated before use rather than folded through arm_nn_size_mul().
    if ((rank <= 0) || (weights_feature_dims->n < 0))
    {
        return -1;
    }

    const int32_t unit_count = weights_feature_dims->n / rank;

    int64_t required_bytes = arm_nn_size_mul(1, input_dims->n);
    required_bytes = arm_nn_size_mul(required_bytes, unit_count);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int32_t));

    return required_bytes;
}

int32_t arm_svdf_s8_input_ctx_get_buffer_size(const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *weights_feature_dims)
{
    return (int32_t)arm_svdf_input_ctx_bytes(input_dims, weights_feature_dims);
}

int32_t arm_svdf_s8_output_ctx_get_buffer_size(const cmsis_nn_svdf_params *svdf_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *weights_feature_dims)
{
    if (svdf_params == NULL)
    {
        return -1;
    }

    return (int32_t)arm_svdf_output_ctx_bytes(svdf_params->rank, input_dims, weights_feature_dims);
}

int32_t arm_svdf_state_s16_s8_input_ctx_get_buffer_size(const cmsis_nn_dims *input_dims,
                                                        const cmsis_nn_dims *weights_feature_dims)
{
    return (int32_t)arm_svdf_input_ctx_bytes(input_dims, weights_feature_dims);
}

int32_t arm_svdf_state_s16_s8_output_ctx_get_buffer_size(const cmsis_nn_svdf_params *svdf_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const cmsis_nn_dims *weights_feature_dims)
{
    if (svdf_params == NULL)
    {
        return -1;
    }

    return (int32_t)arm_svdf_output_ctx_bytes(svdf_params->rank, input_dims, weights_feature_dims);
}

/**
 * @} end of GetBufferSizeSVDF group
 */
