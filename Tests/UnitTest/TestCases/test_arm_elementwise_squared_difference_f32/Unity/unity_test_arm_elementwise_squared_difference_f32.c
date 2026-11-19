/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include <Internal/arm_nn_fp_control.h>
#include "../test_arm_elementwise_squared_difference_f32.c"
#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif
/* Exact reference tests require round-to-nearest and gradual underflow. */
#if defined(__SSE__)
static unsigned saved_fp;
#elif defined(ARM_NN_FP_CONTROL_AVAILABLE)
static arm_nn_fp_control_t saved_fp;
#endif
void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
#if defined(__SSE__)
    __asm volatile("stmxcsr %0" : "=m"(saved_fp));
    unsigned modes = saved_fp & ~0xe040u;
    __asm volatile("ldmxcsr %0" : : "m"(modes));
#elif defined(ARM_NN_FP_CONTROL_AVAILABLE)
    saved_fp = arm_nn_get_fp_control();
    arm_nn_fp_control_t modes = saved_fp & ~(arm_nn_fp_control_t)0x07c80000u;
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2) && defined(ARM_MATH_AUTOVECTORIZE)
    modes |= (arm_nn_fp_control_t)0x01000000u;
#endif
    arm_nn_set_fp_control(modes);
#endif
}
void tearDown(void)
{
#if defined(__SSE__)
    __asm volatile("ldmxcsr %0" : : "m"(saved_fp));
#elif defined(ARM_NN_FP_CONTROL_AVAILABLE)
    arm_nn_set_fp_control(saved_fp);
#endif
}
void test_squared_difference_f32_values(void) { squared_difference_f32_values(); }
void test_squared_difference_f32_alias(void) { squared_difference_f32_alias(); }
void test_squared_difference_f32_flush_to_zero(void) { squared_difference_f32_flush_to_zero(); }
void test_squared_difference_f32_threshold(void) { squared_difference_f32_threshold(); }
void test_squared_difference_f32_arg_error(void) { squared_difference_f32_arg_error(); }
