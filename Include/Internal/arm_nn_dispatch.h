/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Route selection of the s8 wrappers, one copy shared by the wrapper, its scratch query and an ahead-of-time planner
 * (RFC #829). Each selector returns the kernel its wrapper calls for the given layer on the build described by
 * ARM_NN_ROUTE_MVEI, ARM_NN_ROUTE_AUTOVECTORIZE and ARM_NN_TOOLCHAIN_ARMCLANG. The selectors only read dims and params;
 * they do not validate them, which stays with the kernel that is selected.
 */

#ifndef ARM_NN_DISPATCH_H
#define ARM_NN_DISPATCH_H

#include "arm_nnsupportfunctions.h"

/* The build this translation unit compiles kernels for. */
#if defined(ARM_MATH_MVEI)
    #define ARM_NN_BUILD_MVEI 1
#else
    #define ARM_NN_BUILD_MVEI 0
#endif
#if defined(ARM_MATH_AUTOVECTORIZE)
    #define ARM_NN_BUILD_AUTOVECTORIZE 1
#else
    #define ARM_NN_BUILD_AUTOVECTORIZE 0
#endif

/* The build a selector routes for. A library build takes its own; a host planner defines these (0 or 1) to route for a
 * target without compiling that target's intrinsics. */
#ifndef ARM_NN_ROUTE_MVEI
    #define ARM_NN_ROUTE_MVEI ARM_NN_BUILD_MVEI
#endif
#ifndef ARM_NN_ROUTE_AUTOVECTORIZE
    #define ARM_NN_ROUTE_AUTOVECTORIZE ARM_NN_BUILD_AUTOVECTORIZE
#endif

/* A wrapper calls the kernel its selector names, so the two must describe the same build. */
#define ARM_NN_ROUTES_MATCH_BUILD                                                                                      \
    ((ARM_NN_ROUTE_MVEI == ARM_NN_BUILD_MVEI) && (ARM_NN_ROUTE_AUTOVECTORIZE == ARM_NN_BUILD_AUTOVECTORIZE))

/** The kernel arm_convolve_wrapper_s8() calls. */
typedef enum
{
    ARM_NN_CONV_S8_ROUTE_1X1_FAST = 0,   /**< arm_convolve_1x1_s8_fast() */
    ARM_NN_CONV_S8_ROUTE_1X1 = 1,        /**< arm_convolve_1x1_s8() */
    ARM_NN_CONV_S8_ROUTE_1_X_N = 2,      /**< arm_convolve_1_x_n_s8() */
    ARM_NN_CONV_S8_ROUTE_1X1_OUT = 3,    /**< arm_convolve_1x1_out_s8() (MVE) */
    ARM_NN_CONV_S8_ROUTE_SMALL_CIN = 4,  /**< arm_convolve_s8_small_cin() (MVE, not AUTOVECTORIZE) */
    ARM_NN_CONV_S8_ROUTE_3X3_C16_S1 = 5, /**< arm_convolve_s8_3x3_c16_s1() (MVE, not AUTOVECTORIZE) */
    ARM_NN_CONV_S8_ROUTE_GENERIC = 6     /**< arm_convolve_s8() */
} arm_nn_conv_s8_route;

/** The kernel arm_depthwise_conv_wrapper_s8() calls. */
typedef enum
{
    ARM_NN_DW_CONV_S8_ROUTE_TO_CONV = 0, /**< arm_depthwise_conv_to_conv_s8() (MVE, one input channel) */
    ARM_NN_DW_CONV_S8_ROUTE_3X3 = 1,     /**< arm_depthwise_conv_3x3_s8() (not MVE) */
    ARM_NN_DW_CONV_S8_ROUTE_OPT = 2,     /**< arm_depthwise_conv_s8_opt() */
    ARM_NN_DW_CONV_S8_ROUTE_GENERIC = 3  /**< arm_depthwise_conv_s8() */
} arm_nn_dw_conv_s8_route;

/** The kernel arm_fully_connected_wrapper_s8() calls. */
typedef enum
{
    ARM_NN_FC_S8_ROUTE_PER_CHANNEL = 0, /**< arm_fully_connected_per_channel_s8() */
    ARM_NN_FC_S8_ROUTE_PER_TENSOR = 1   /**< arm_fully_connected_s8() */
} arm_nn_fc_s8_route;

__STATIC_FORCEINLINE arm_nn_conv_s8_route arm_nn_convolve_wrapper_s8_select(const cmsis_nn_conv_params *conv_params,
                                                                            const cmsis_nn_dims *input_dims,
                                                                            const cmsis_nn_dims *filter_dims,
                                                                            const cmsis_nn_dims *output_dims)
{
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        return arm_nn_is_convolve_1x1_fast(conv_params) ? ARM_NN_CONV_S8_ROUTE_1X1_FAST : ARM_NN_CONV_S8_ROUTE_1X1;
    }
    if (arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims) &&
        arm_nn_convolve_1_x_n_s8_padding_supported(conv_params, filter_dims, output_dims))
    {
        return ARM_NN_CONV_S8_ROUTE_1_X_N;
    }
#if ARM_NN_ROUTE_MVEI
    // MVE optimization of the Bx1x1xC case.
    if ((output_dims->h == 1) && (output_dims->w == 1) && (((int64_t)conv_params->stride.w * input_dims->c) % 4 == 0) &&
        (input_dims->c == filter_dims->c))
    {
        return ARM_NN_CONV_S8_ROUTE_1X1_OUT;
    }
#endif
#if ARM_NN_ROUTE_MVEI && !ARM_NN_ROUTE_AUTOVECTORIZE
    /* Layers in the gate of a direct entry run it, with the same result and scratch as arm_convolve_s8(). The input
       depth rejects most layers before the rest of either gate is read. */
    if (input_dims->c <= 3)
    {
        if (arm_nn_is_convolve_s8_small_cin(conv_params, input_dims, filter_dims, output_dims, NULL))
        {
            return ARM_NN_CONV_S8_ROUTE_SMALL_CIN;
        }
    }
    else if (input_dims->c == 16)
    {
        if (arm_nn_is_convolve_s8_3x3_c16_s1(conv_params, input_dims, filter_dims, NULL))
        {
            return ARM_NN_CONV_S8_ROUTE_3X3_C16_S1;
        }
    }
#endif
    return ARM_NN_CONV_S8_ROUTE_GENERIC;
}

__STATIC_FORCEINLINE arm_nn_dw_conv_s8_route
arm_nn_depthwise_conv_wrapper_s8_select(const cmsis_nn_dw_conv_params *dw_conv_params,
                                        const cmsis_nn_dims *input_dims,
                                        const cmsis_nn_dims *filter_dims,
                                        const cmsis_nn_dims *output_dims)
{
#if ARM_NN_ROUTE_MVEI
    if (input_dims->c == 1 && output_dims->c > CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD)
    {
        return ARM_NN_DW_CONV_S8_ROUTE_TO_CONV;
    }
#endif
    if (1 == dw_conv_params->ch_mult && input_dims->n == 1 &&
        arm_nn_dw_conv_opt_dilation_supported(dw_conv_params, input_dims, filter_dims, output_dims))
    {
#if !ARM_NN_ROUTE_MVEI
        if (filter_dims->w == 3 && filter_dims->h == 3 && dw_conv_params->padding.h <= 1 &&
            dw_conv_params->padding.w <= 1 && dw_conv_params->dilation.w == 1 && dw_conv_params->dilation.h == 1)
        {
            return ARM_NN_DW_CONV_S8_ROUTE_3X3;
        }
#endif
        return ARM_NN_DW_CONV_S8_ROUTE_OPT;
    }
    return ARM_NN_DW_CONV_S8_ROUTE_GENERIC;
}

__STATIC_FORCEINLINE arm_nn_fc_s8_route
arm_nn_fully_connected_wrapper_s8_select(const cmsis_nn_quant_params *quant_params)
{
    return quant_params->is_per_channel ? ARM_NN_FC_S8_ROUTE_PER_CHANNEL : ARM_NN_FC_S8_ROUTE_PER_TENSOR;
}

#endif
