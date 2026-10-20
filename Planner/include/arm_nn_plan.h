/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Ahead-of-time planning for ns-cmsis-nn layers (RFC #829), built for the host.
 *
 * A plan answers, for one layer on one target, which kernel a call reaches and what the caller must provide for it:
 * the size and role of the ctx buffer and the size of any prepared constant stream. It uses the library's own route
 * selectors (Include/Internal/arm_nn_dispatch.h) and scratch-size queries, so a planner and the firmware cannot
 * disagree. The ISA, autovectorization and rounding of the target are runtime inputs; the toolchain is fixed when
 * this library is built (a few routes and sizes depend on it at compile time) and a target naming another toolchain
 * is rejected.
 *
 * Every function validates its arguments and returns a status; a plan is written only on ARM_NN_PLAN_OK. Nothing here
 * allocates, and no size is computed in a type that can overflow.
 */

#ifndef ARM_NN_PLAN_H
#define ARM_NN_PLAN_H

#include "arm_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Version of the functions and structures below; bumped on any incompatible change. */
#define ARM_NN_PLAN_ABI_VERSION 1

/** Instruction set the target library is built for. */
typedef enum
{
    ARM_NN_PLAN_ISA_SCALAR = 0, /**< neither ARM_MATH_DSP nor ARM_MATH_MVEI (Cortex-M0/M0+) */
    ARM_NN_PLAN_ISA_DSP = 1,    /**< ARM_MATH_DSP without ARM_MATH_MVEI (Cortex-M4/M7/M33) */
    ARM_NN_PLAN_ISA_MVE = 2     /**< ARM_MATH_MVEI (Cortex-M55/M85) */
} arm_nn_plan_isa;

/** Compiler the target library is built with. */
typedef enum
{
    ARM_NN_PLAN_TOOLCHAIN_GCC = 0,     /**< GCC, or any compiler that is not Arm Compiler 6 */
    ARM_NN_PLAN_TOOLCHAIN_ARMCLANG = 1 /**< Arm Compiler 6 */
} arm_nn_plan_toolchain;

/** The build of the target library a plan is for. */
typedef struct
{
    int32_t isa;             /**< arm_nn_plan_isa */
    int32_t autovectorize;   /**< 1 when the library is built with ARM_MATH_AUTOVECTORIZE (MVE only), else 0 */
    int32_t single_rounding; /**< 1 when the library is built with CMSIS_NN_USE_SINGLE_ROUNDING, else 0 */
    int32_t toolchain;       /**< arm_nn_plan_toolchain; must equal arm_nn_plan_toolchain_built() */
} arm_nn_plan_target;

/** Outcome of a plan. */
typedef enum
{
    ARM_NN_PLAN_OK = 0,
    ARM_NN_PLAN_E_NULL = 1,       /**< a required pointer is NULL */
    ARM_NN_PLAN_E_TARGET = 2,     /**< a target field is out of range, or names a toolchain this library is not for */
    ARM_NN_PLAN_E_SHAPE = 3,      /**< a dimension is not positive, or the dims and params disagree */
    ARM_NN_PLAN_E_RANGE = 4,      /**< a tensor or buffer size does not fit int32_t */
    ARM_NN_PLAN_E_UNSUPPORTED = 5 /**< the entry declines this layer or target (ARM_CMSIS_NN_NO_IMPL_ERROR) */
} arm_nn_plan_status;

/** What the ctx argument of the planned call holds. */
typedef enum
{
    ARM_NN_PLAN_CTX_NONE = 0,       /**< not read; ctx->buf may be NULL */
    ARM_NN_PLAN_CTX_SCRATCH = 1,    /**< scratch, contents undefined between calls */
    ARM_NN_PLAN_CTX_KERNEL_SUMS = 2 /**< prepared once with arm_vector_sum_s8() and kept for every call */
} arm_nn_plan_ctx_role;

/** The plan of one call. */
typedef struct
{
    const char *kernel;     /**< name of the kernel the call reaches (static string, not owned by the caller) */
    const char *route_id;   /**< the route's id in Tests/KernelContracts/kernel_implementations.json, or NULL for a
                                 direct entry (static string) */
    int32_t route;          /**< the wrapper's route (Include/Internal/arm_nn_dispatch.h), or -1 for a direct entry */
    int32_t ctx_role;       /**< arm_nn_plan_ctx_role */
    int32_t ctx_bytes;      /**< bytes ctx->buf must hold; 0 with ARM_NN_PLAN_CTX_NONE */
    int32_t prepared_bytes; /**< bytes of a prepared stream that replaces the filter data, else 0 */
} arm_nn_plan_result;

/** ARM_NN_PLAN_ABI_VERSION of the library. */
int32_t arm_nn_plan_abi_version(void);

/** The arm_nn_plan_toolchain this library plans for. */
int32_t arm_nn_plan_toolchain_built(void);

/**
 * @brief Plan arm_depthwise_conv_wrapper_s8().
 *
 * Dims: input [N, H, W, C_IN], filter [1, H, W, C_OUT], output [N, H, W, C_OUT] with C_OUT = C_IN * ch_mult.
 * ctx is scratch sized as arm_depthwise_conv_wrapper_s8_get_buffer_size() for the target.
 */
arm_nn_plan_status arm_nn_plan_depthwise_conv_wrapper_s8(const arm_nn_plan_target *target,
                                                         const cmsis_nn_dw_conv_params *dw_conv_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const cmsis_nn_dims *output_dims,
                                                         arm_nn_plan_result *plan);

/**
 * @brief Plan arm_fully_connected_wrapper_s8().
 *
 * Dims: input [N, H, W, C_IN] (accumulation depth H * W * C_IN), filter [N = accumulation depth, C = output depth],
 * output [N, C_OUT]. On MVE ctx holds the kernel sums; elsewhere it is not read.
 */
arm_nn_plan_status arm_nn_plan_fully_connected_wrapper_s8(const arm_nn_plan_target *target,
                                                          const cmsis_nn_quant_params *quant_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims,
                                                          arm_nn_plan_result *plan);

/**
 * @brief Plan arm_fully_connected_per_channel_packed_s8(), which no wrapper reaches.
 *
 * The layer must pass arm_nn_fc_packed_s8_supported() and the target must be MVE without ARM_MATH_AUTOVECTORIZE or
 * CMSIS_NN_USE_SINGLE_ROUNDING, else ARM_NN_PLAN_E_UNSUPPORTED. The plan's prepared_bytes is the stream
 * arm_fully_connected_per_channel_packed_s8_pack() writes; ctx is not read.
 */
arm_nn_plan_status
arm_nn_plan_fully_connected_per_channel_packed_s8(const arm_nn_plan_target *target,
                                                  const cmsis_nn_fc_params *fc_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *filter_dims,
                                                  arm_nn_plan_result *plan);

/**
 * @brief Plan arm_avgpool_s8() (and arm_max_pool_s8() when max is 1).
 *
 * Dims: input [N, H, W, C], filter [-, H, W, -], output [N, H, W, C]. avgpool scratch is sized as
 * arm_avgpool_s8_get_buffer_size() for the target; max pooling reads no ctx.
 */
arm_nn_plan_status arm_nn_plan_pool_s8(const arm_nn_plan_target *target,
                                       int32_t max,
                                       const cmsis_nn_pool_params *pool_params,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *filter_dims,
                                       const cmsis_nn_dims *output_dims,
                                       arm_nn_plan_result *plan);

#ifdef __cplusplus
}
#endif

#endif
