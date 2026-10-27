/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* The planner routes for targets through ARM_NN_ROUTE_*; compiled for a target it would size for its own build. */
#if defined(ARM_MATH_DSP) || defined(ARM_MATH_MVEI) || defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
    #error "the ns-cmsis-nn planner is a host library: build it without ARM_MATH_DSP or ARM_MATH_MVEI"
#endif

#include "arm_nn_plan.h"

#include "Internal/arm_nn_compiler.h"
#include "Internal/arm_nn_dispatch.h"
#include "arm_nn_plan_internal.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <string.h>

int32_t arm_nn_plan_abi_version(void) { return ARM_NN_PLAN_ABI_VERSION; }

int32_t arm_nn_plan_toolchain_built(void)
{
    return ARM_NN_TOOLCHAIN_ARMCLANG ? ARM_NN_PLAN_TOOLCHAIN_ARMCLANG : ARM_NN_PLAN_TOOLCHAIN_GCC;
}

static arm_nn_plan_status check_target(const arm_nn_plan_target *target)
{
    if (target == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    if (target->isa < ARM_NN_PLAN_ISA_SCALAR || target->isa > ARM_NN_PLAN_ISA_MVE ||
        (target->autovectorize != 0 && target->autovectorize != 1) ||
        (target->single_rounding != 0 && target->single_rounding != 1) ||
        (target->autovectorize && target->isa != ARM_NN_PLAN_ISA_MVE) ||
        target->toolchain != arm_nn_plan_toolchain_built())
    {
        return ARM_NN_PLAN_E_TARGET;
    }
    return ARM_NN_PLAN_OK;
}

/* Every dimension positive and the element count within int32_t. */
static arm_nn_plan_status check_dims(const cmsis_nn_dims *dims)
{
    if (dims == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    if (dims->n < 1 || dims->h < 1 || dims->w < 1 || dims->c < 1)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }
    const int64_t count = arm_nn_size_mul(arm_nn_size_mul(arm_nn_size_mul(dims->n, dims->h), dims->w), dims->c);
    return count < 0 ? ARM_NN_PLAN_E_RANGE : ARM_NN_PLAN_OK;
}

#define PLAN_TRY(expr)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const arm_nn_plan_status plan_try_status = (expr);                                                             \
        if (plan_try_status != ARM_NN_PLAN_OK)                                                                         \
        {                                                                                                              \
            return plan_try_status;                                                                                    \
        }                                                                                                              \
    } while (0)

static arm_nn_plan_status
check_window(const cmsis_nn_tile *stride, const cmsis_nn_tile *padding, const cmsis_nn_tile *dilation)
{
    if (stride->w < 1 || stride->h < 1 || padding->w < 0 || padding->h < 0 ||
        (dilation != NULL && (dilation->w < 1 || dilation->h < 1)))
    {
        return ARM_NN_PLAN_E_SHAPE;
    }
    return ARM_NN_PLAN_OK;
}

/* A depthwise layer: dims and window valid, C_OUT = C_IN * ch_mult, a single filter batch and matching batches. */
static arm_nn_plan_status check_dw_layer(const cmsis_nn_dw_conv_params *dw_conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims)
{
    PLAN_TRY(check_dims(input_dims));
    PLAN_TRY(check_dims(filter_dims));
    PLAN_TRY(check_dims(output_dims));
    PLAN_TRY(check_window(&dw_conv_params->stride, &dw_conv_params->padding, &dw_conv_params->dilation));
    if (dw_conv_params->ch_mult < 1 || (int64_t)input_dims->c * dw_conv_params->ch_mult != output_dims->c ||
        filter_dims->n != 1 || filter_dims->c != output_dims->c || output_dims->n != input_dims->n)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }
    return ARM_NN_PLAN_OK;
}

/* A scratch-size query's answer: a negative one is the sizers' out-of-range sentinel. */
static arm_nn_plan_status write_plan(arm_nn_plan_result *plan,
                                     const char *kernel,
                                     const char *route_id,
                                     int32_t route,
                                     int32_t ctx_bytes,
                                     int32_t ctx_role_when_used,
                                     int32_t prepared_bytes)
{
    if (ctx_bytes < 0 || prepared_bytes < 0)
    {
        return ARM_NN_PLAN_E_RANGE;
    }
    plan->kernel = kernel;
    plan->route_id = route_id;
    plan->route = route;
    plan->ctx_role = ctx_bytes > 0 ? ctx_role_when_used : ARM_NN_PLAN_CTX_NONE;
    plan->ctx_bytes = ctx_bytes;
    plan->prepared_bytes = prepared_bytes;
    return ARM_NN_PLAN_OK;
}

arm_nn_plan_status arm_nn_plan_depthwise_conv_wrapper_s8(const arm_nn_plan_target *target,
                                                         const cmsis_nn_dw_conv_params *dw_conv_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const cmsis_nn_dims *output_dims,
                                                         arm_nn_plan_result *plan)
{
    if (dw_conv_params == NULL || plan == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    PLAN_TRY(check_target(target));
    PLAN_TRY(check_dw_layer(dw_conv_params, input_dims, filter_dims, output_dims));

    int32_t route;
    int32_t ctx_bytes;
    switch (target->isa)
    {
    case ARM_NN_PLAN_ISA_MVE:
        route = target->autovectorize
            ? arm_nn_plan_route_dw_conv_s8_mve_autovec(dw_conv_params, input_dims, filter_dims, output_dims)
            : arm_nn_plan_route_dw_conv_s8_mve(dw_conv_params, input_dims, filter_dims, output_dims);
        ctx_bytes =
            arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(dw_conv_params, input_dims, filter_dims, output_dims);
        break;
    case ARM_NN_PLAN_ISA_DSP:
        route = arm_nn_plan_route_dw_conv_s8_scalar(dw_conv_params, input_dims, filter_dims, output_dims);
        ctx_bytes =
            arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(dw_conv_params, input_dims, filter_dims, output_dims);
        break;
    default:
        /* This library is a scalar host build, so the dispatching query takes the scalar leg. */
        route = arm_nn_plan_route_dw_conv_s8_scalar(dw_conv_params, input_dims, filter_dims, output_dims);
        ctx_bytes = arm_depthwise_conv_wrapper_s8_get_buffer_size(dw_conv_params, input_dims, filter_dims, output_dims);
        break;
    }

    static const char *const kernels[] = {
        [ARM_NN_DW_CONV_S8_ROUTE_TO_CONV] = "arm_depthwise_conv_to_conv_s8",
        [ARM_NN_DW_CONV_S8_ROUTE_3X3] = "arm_depthwise_conv_3x3_s8",
        [ARM_NN_DW_CONV_S8_ROUTE_OPT] = "arm_depthwise_conv_s8_opt",
        [ARM_NN_DW_CONV_S8_ROUTE_GENERIC] = "arm_depthwise_conv_s8",
    };
    static const char *const route_ids[] = {
        [ARM_NN_DW_CONV_S8_ROUTE_TO_CONV] = "dw_wrapper_s8:to_conv",
        [ARM_NN_DW_CONV_S8_ROUTE_3X3] = "dw_wrapper_s8:conv_3x3",
        [ARM_NN_DW_CONV_S8_ROUTE_OPT] = "dw_wrapper_s8:opt",
        [ARM_NN_DW_CONV_S8_ROUTE_GENERIC] = "dw_wrapper_s8:generic",
    };
    if (route < 0 || route > ARM_NN_DW_CONV_S8_ROUTE_GENERIC)
    {
        return ARM_NN_PLAN_E_UNSUPPORTED;
    }
    return write_plan(plan, kernels[route], route_ids[route], route, ctx_bytes, ARM_NN_PLAN_CTX_SCRATCH, 0);
}

static int32_t dw_opt_scratch_bytes(const arm_nn_plan_target *target,
                                    const cmsis_nn_dims *input_dims,
                                    const cmsis_nn_dims *filter_dims)
{
    switch (target->isa)
    {
    case ARM_NN_PLAN_ISA_MVE:
        return arm_depthwise_conv_s8_opt_get_buffer_size_mve(input_dims, filter_dims);
    case ARM_NN_PLAN_ISA_DSP:
        return arm_depthwise_conv_s8_opt_get_buffer_size_dsp(input_dims, filter_dims);
    default:
        return arm_depthwise_conv_s8_opt_get_buffer_size(input_dims, filter_dims);
    }
}

arm_nn_plan_status arm_nn_plan_depthwise_conv_s8_opt(const arm_nn_plan_target *target,
                                                     const cmsis_nn_dw_conv_params *dw_conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims,
                                                     int32_t ctx_size,
                                                     arm_nn_plan_result *plan)
{
    if (dw_conv_params == NULL || plan == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    PLAN_TRY(check_target(target));
    PLAN_TRY(check_dw_layer(dw_conv_params, input_dims, filter_dims, output_dims));
    /* arm_depthwise_conv_s8_opt_channelwise() returns ARM_CMSIS_NN_ARG_ERROR for these on every build. */
    if (dw_conv_params->ch_mult != 1 || dw_conv_params->dilation.h != 1)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }
    if (ctx_size < 0)
    {
        return ARM_NN_PLAN_E_RANGE;
    }
    const int32_t ctx_bytes = dw_opt_scratch_bytes(target, input_dims, filter_dims);
    if (ctx_bytes < 0)
    {
        return ARM_NN_PLAN_E_RANGE;
    }

    const int32_t route = target->isa == ARM_NN_PLAN_ISA_MVE
        ? (target->autovectorize
               ? arm_nn_plan_route_dw_opt_s8_mve_autovec(dw_conv_params, input_dims, filter_dims, output_dims, ctx_size)
               : arm_nn_plan_route_dw_opt_s8_mve(dw_conv_params, input_dims, filter_dims, output_dims, ctx_size))
        : arm_nn_plan_route_dw_opt_s8_scalar(dw_conv_params, input_dims, filter_dims, output_dims, ctx_size);
    if (route == ARM_NN_DW_OPT_S8_ROUTE_PLANAR)
    {
        return write_plan(
            plan, "arm_depthwise_conv_s8_opt_planar", "dw_opt_s8:planar", route, ctx_bytes, ARM_NN_PLAN_CTX_SCRATCH, 0);
    }
    if (route != ARM_NN_DW_OPT_S8_ROUTE_CHANNELWISE)
    {
        return ARM_NN_PLAN_E_UNSUPPORTED;
    }
    /* The channelwise path rejects a declared ctx->size below its scratch; the DSP path also counts channels and taps
       in uint16_t. */
    if (ctx_size != 0 && ctx_size < ctx_bytes)
    {
        return ARM_NN_PLAN_E_RANGE;
    }
    if (target->isa == ARM_NN_PLAN_ISA_DSP &&
        (input_dims->c > UINT16_MAX || (int64_t)filter_dims->w * filter_dims->h > UINT16_MAX))
    {
        return ARM_NN_PLAN_E_RANGE;
    }
    return write_plan(plan,
                      "arm_depthwise_conv_s8_opt_channelwise",
                      "dw_opt_s8:channelwise",
                      route,
                      ctx_bytes,
                      ARM_NN_PLAN_CTX_SCRATCH,
                      0);
}

arm_nn_plan_status arm_nn_plan_fully_connected_wrapper_s8(const arm_nn_plan_target *target,
                                                          const cmsis_nn_quant_params *quant_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims,
                                                          arm_nn_plan_result *plan)
{
    if (quant_params == NULL || plan == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    PLAN_TRY(check_target(target));
    PLAN_TRY(check_dims(input_dims));
    PLAN_TRY(check_dims(filter_dims));
    PLAN_TRY(check_dims(output_dims));
    const int64_t accumulation = (int64_t)input_dims->h * input_dims->w * input_dims->c;
    if (accumulation != filter_dims->n || output_dims->c != filter_dims->c || output_dims->n != input_dims->n)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }

    int32_t route;
    switch (target->isa)
    {
    case ARM_NN_PLAN_ISA_MVE:
        route = target->autovectorize ? arm_nn_plan_route_fc_s8_mve_autovec(quant_params)
                                      : arm_nn_plan_route_fc_s8_mve(quant_params);
        break;
    default:
        route = arm_nn_plan_route_fc_s8_scalar(quant_params);
        break;
    }
    /* The MVE kernels read precomputed kernel sums from ctx; the other legs add the bias themselves. */
    const int32_t ctx_bytes = target->isa == ARM_NN_PLAN_ISA_MVE
        ? arm_fully_connected_s8_get_buffer_size_mve(filter_dims)
        : arm_fully_connected_s8_get_buffer_size_dsp(filter_dims);
    return write_plan(
        plan,
        route == ARM_NN_FC_S8_ROUTE_PER_CHANNEL ? "arm_fully_connected_per_channel_s8" : "arm_fully_connected_s8",
        route == ARM_NN_FC_S8_ROUTE_PER_CHANNEL ? "fc_wrapper_s8:per_channel" : "fc_wrapper_s8:per_tensor",
        route,
        ctx_bytes,
        ARM_NN_PLAN_CTX_KERNEL_SUMS,
        0);
}

arm_nn_plan_status
arm_nn_plan_fully_connected_per_channel_packed_s8(const arm_nn_plan_target *target,
                                                  const cmsis_nn_fc_params *fc_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *filter_dims,
                                                  arm_nn_plan_result *plan)
{
    if (fc_params == NULL || quant_params == NULL || plan == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    PLAN_TRY(check_target(target));
    PLAN_TRY(check_dims(filter_dims));
    if (quant_params->multiplier == NULL || quant_params->shift == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    if (arm_nn_fc_packed_s8_size(filter_dims->n, filter_dims->c) > INT32_MAX)
    {
        return ARM_NN_PLAN_E_RANGE;
    }
    /* The entry returns ARM_CMSIS_NN_NO_IMPL_ERROR off this build. */
    if (target->isa != ARM_NN_PLAN_ISA_MVE || target->autovectorize || target->single_rounding ||
        !arm_nn_fc_packed_s8_supported(fc_params, quant_params, filter_dims))
    {
        return ARM_NN_PLAN_E_UNSUPPORTED;
    }
    return write_plan(plan,
                      "arm_fully_connected_per_channel_packed_s8",
                      NULL,
                      -1,
                      0,
                      ARM_NN_PLAN_CTX_NONE,
                      arm_fully_connected_per_channel_packed_s8_get_packed_size(filter_dims));
}

arm_nn_plan_status arm_nn_plan_pool_s8(const arm_nn_plan_target *target,
                                       int32_t max,
                                       const cmsis_nn_pool_params *pool_params,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *filter_dims,
                                       const cmsis_nn_dims *output_dims,
                                       arm_nn_plan_result *plan)
{
    if (pool_params == NULL || filter_dims == NULL || plan == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    PLAN_TRY(check_target(target));
    if (max != 0 && max != 1)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }
    PLAN_TRY(check_dims(input_dims));
    PLAN_TRY(check_dims(output_dims));
    PLAN_TRY(check_window(&pool_params->stride, &pool_params->padding, NULL));
    if (filter_dims->h < 1 || filter_dims->w < 1 || output_dims->c != input_dims->c || output_dims->n != input_dims->n)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }
    if (max)
    {
        return write_plan(plan, "arm_max_pool_s8", NULL, -1, 0, ARM_NN_PLAN_CTX_NONE, 0);
    }

    int32_t ctx_bytes;
    switch (target->isa)
    {
    case ARM_NN_PLAN_ISA_MVE:
        ctx_bytes = arm_avgpool_s8_get_buffer_size_mve(output_dims->w, input_dims->c);
        break;
    case ARM_NN_PLAN_ISA_DSP:
        ctx_bytes = arm_avgpool_s8_get_buffer_size_dsp(output_dims->w, input_dims->c);
        break;
    default:
        ctx_bytes = arm_avgpool_s8_get_buffer_size(output_dims->w, input_dims->c);
        break;
    }
    return write_plan(plan, "arm_avgpool_s8", NULL, -1, ctx_bytes, ARM_NN_PLAN_CTX_SCRATCH, 0);
}

typedef enum
{
    SIZE_ARG_DW_CONV_PARAMS = 1 << 0,
    SIZE_ARG_INPUT_DIMS = 1 << 1,
    SIZE_ARG_FILTER_DIMS = 1 << 2,
    SIZE_ARG_OUTPUT_DIMS = 1 << 3,
    SIZE_ARG_DIM_DST_WIDTH = 1 << 4,
    SIZE_ARG_CH_SRC = 1 << 5
} size_arg;

typedef enum
{
    SIZE_AVGPOOL_S8,
    SIZE_CONV_S8_WEIGHTS_SUM,
    SIZE_DW_OPT_S8,
    SIZE_DW_OPT_3X3_S8,
    SIZE_DW_WRAPPER_S8,
    SIZE_FC_S8,
    SIZE_FC_PACKED_S8
} size_query_id;

static const struct
{
    const char *symbol;
    size_query_id id;
    uint32_t args;
} size_queries[] = {
    {"arm_avgpool_s8_get_buffer_size", SIZE_AVGPOOL_S8, SIZE_ARG_DIM_DST_WIDTH | SIZE_ARG_CH_SRC},
    {"arm_convolve_s8_get_weights_sum_size", SIZE_CONV_S8_WEIGHTS_SUM, SIZE_ARG_OUTPUT_DIMS},
    {"arm_depthwise_conv_s8_opt_get_buffer_size", SIZE_DW_OPT_S8, SIZE_ARG_INPUT_DIMS | SIZE_ARG_FILTER_DIMS},
    {"arm_depthwise_conv_s8_opt_3x3_get_buffer_size", SIZE_DW_OPT_3X3_S8, SIZE_ARG_INPUT_DIMS},
    {"arm_depthwise_conv_wrapper_s8_get_buffer_size",
     SIZE_DW_WRAPPER_S8,
     SIZE_ARG_DW_CONV_PARAMS | SIZE_ARG_INPUT_DIMS | SIZE_ARG_FILTER_DIMS | SIZE_ARG_OUTPUT_DIMS},
    {"arm_fully_connected_s8_get_buffer_size", SIZE_FC_S8, SIZE_ARG_FILTER_DIMS},
    {"arm_fully_connected_per_channel_packed_s8_get_packed_size", SIZE_FC_PACKED_S8, SIZE_ARG_FILTER_DIMS},
};

static int32_t
size_query_answer(const arm_nn_plan_target *target, size_query_id id, const arm_nn_plan_size_operands *op)
{
    const int32_t isa = target->isa;
    switch (id)
    {
    case SIZE_AVGPOOL_S8:
        return isa == ARM_NN_PLAN_ISA_MVE ? arm_avgpool_s8_get_buffer_size_mve(*op->dim_dst_width, *op->ch_src)
            : isa == ARM_NN_PLAN_ISA_DSP  ? arm_avgpool_s8_get_buffer_size_dsp(*op->dim_dst_width, *op->ch_src)
                                          : arm_avgpool_s8_get_buffer_size(*op->dim_dst_width, *op->ch_src);
    case SIZE_CONV_S8_WEIGHTS_SUM:
        return isa == ARM_NN_PLAN_ISA_MVE ? arm_nn_plan_route_weights_sum_bytes_mve(op->output_dims)
                                          : arm_nn_plan_route_weights_sum_bytes_scalar(op->output_dims);
    case SIZE_DW_OPT_S8:
        return dw_opt_scratch_bytes(target, op->input_dims, op->filter_dims);
    case SIZE_DW_OPT_3X3_S8:
        return arm_depthwise_conv_s8_opt_3x3_get_buffer_size(op->input_dims);
    case SIZE_DW_WRAPPER_S8:
        return isa == ARM_NN_PLAN_ISA_MVE ? arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(
                                                op->dw_conv_params, op->input_dims, op->filter_dims, op->output_dims)
            : isa == ARM_NN_PLAN_ISA_DSP ? arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(
                                               op->dw_conv_params, op->input_dims, op->filter_dims, op->output_dims)
                                         : arm_depthwise_conv_wrapper_s8_get_buffer_size(
                                               op->dw_conv_params, op->input_dims, op->filter_dims, op->output_dims);
    case SIZE_FC_S8:
        return isa == ARM_NN_PLAN_ISA_MVE ? arm_fully_connected_s8_get_buffer_size_mve(op->filter_dims)
            : isa == ARM_NN_PLAN_ISA_DSP  ? arm_fully_connected_s8_get_buffer_size_dsp(op->filter_dims)
                                          : arm_fully_connected_s8_get_buffer_size(op->filter_dims);
    case SIZE_FC_PACKED_S8:
        /* The query answers 0, not -1, past int32_t. */
        if (arm_nn_fc_packed_s8_size(op->filter_dims->n, op->filter_dims->c) > INT32_MAX)
        {
            return -1;
        }
        return arm_fully_connected_per_channel_packed_s8_get_packed_size(op->filter_dims);
    }
    return -1;
}

arm_nn_plan_status arm_nn_plan_buffer_bytes(const arm_nn_plan_target *target,
                                            const char *size_query,
                                            const arm_nn_plan_size_operands *operands,
                                            int32_t *bytes)
{
    if (size_query == NULL || operands == NULL || bytes == NULL)
    {
        return ARM_NN_PLAN_E_NULL;
    }
    PLAN_TRY(check_target(target));
    size_t query = 0;
    while (query < sizeof(size_queries) / sizeof(size_queries[0]) &&
           strcmp(size_queries[query].symbol, size_query) != 0)
    {
        query++;
    }
    if (query == sizeof(size_queries) / sizeof(size_queries[0]))
    {
        return ARM_NN_PLAN_E_UNSUPPORTED;
    }

    const uint32_t takes = size_queries[query].args;
    const struct
    {
        uint32_t arg;
        const void *value;
    } given[] = {
        {SIZE_ARG_DW_CONV_PARAMS, operands->dw_conv_params},
        {SIZE_ARG_INPUT_DIMS, operands->input_dims},
        {SIZE_ARG_FILTER_DIMS, operands->filter_dims},
        {SIZE_ARG_OUTPUT_DIMS, operands->output_dims},
        {SIZE_ARG_DIM_DST_WIDTH, operands->dim_dst_width},
        {SIZE_ARG_CH_SRC, operands->ch_src},
    };
    for (size_t i = 0; i < sizeof(given) / sizeof(given[0]); i++)
    {
        if ((takes & given[i].arg) && given[i].value == NULL)
        {
            return ARM_NN_PLAN_E_NULL;
        }
        if (!(takes & given[i].arg) && given[i].value != NULL)
        {
            return ARM_NN_PLAN_E_SHAPE;
        }
    }

    if (takes & SIZE_ARG_DW_CONV_PARAMS)
    {
        PLAN_TRY(check_dw_layer(
            operands->dw_conv_params, operands->input_dims, operands->filter_dims, operands->output_dims));
    }
    else
    {
        if (takes & SIZE_ARG_INPUT_DIMS)
        {
            PLAN_TRY(check_dims(operands->input_dims));
        }
        if (takes & SIZE_ARG_FILTER_DIMS)
        {
            PLAN_TRY(check_dims(operands->filter_dims));
        }
        if (takes & SIZE_ARG_OUTPUT_DIMS)
        {
            PLAN_TRY(check_dims(operands->output_dims));
        }
    }
    if (((takes & SIZE_ARG_DIM_DST_WIDTH) && *operands->dim_dst_width < 1) ||
        ((takes & SIZE_ARG_CH_SRC) && *operands->ch_src < 1))
    {
        return ARM_NN_PLAN_E_SHAPE;
    }

    const int32_t answer = size_query_answer(target, size_queries[query].id, operands);
    if (answer < 0)
    {
        return ARM_NN_PLAN_E_RANGE;
    }
    *bytes = answer;
    return ARM_NN_PLAN_OK;
}
