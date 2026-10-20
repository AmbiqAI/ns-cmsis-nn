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
    const int64_t count = (int64_t)dims->n * dims->h * dims->w * dims->c;
    return count > INT32_MAX ? ARM_NN_PLAN_E_RANGE : ARM_NN_PLAN_OK;
}

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

#define PLAN_TRY(expr)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const arm_nn_plan_status plan_try_status = (expr);                                                             \
        if (plan_try_status != ARM_NN_PLAN_OK)                                                                         \
        {                                                                                                              \
            return plan_try_status;                                                                                    \
        }                                                                                                              \
    } while (0)

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
    PLAN_TRY(check_dims(input_dims));
    PLAN_TRY(check_dims(filter_dims));
    PLAN_TRY(check_dims(output_dims));
    PLAN_TRY(check_window(&dw_conv_params->stride, &dw_conv_params->padding, &dw_conv_params->dilation));
    if (dw_conv_params->ch_mult < 1 || (int64_t)input_dims->c * dw_conv_params->ch_mult != output_dims->c ||
        filter_dims->n != 1 || filter_dims->c != output_dims->c || output_dims->n != input_dims->n)
    {
        return ARM_NN_PLAN_E_SHAPE;
    }

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
