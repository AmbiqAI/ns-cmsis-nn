/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "../test_arm_elementwise_div_f32.c"
#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif
/* Exact reference tests require round-to-nearest and gradual underflow. */
#if defined(__SSE__)
static unsigned saved_fp;
#elif defined(__ARM_FP) && (__ARM_FP != 0)
static unsigned saved_fp;
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
#elif defined(__ARM_FP) && (__ARM_FP != 0)
    __asm volatile("vmrs %0, fpscr" : "=r"(saved_fp));
    unsigned modes = saved_fp & ~0x07c80000u;
    __asm volatile("vmsr fpscr, %0" : : "r"(modes) : "memory");
#endif
}
void tearDown(void)
{
#if defined(__SSE__)
    __asm volatile("ldmxcsr %0" : : "m"(saved_fp));
#elif defined(__ARM_FP) && (__ARM_FP != 0)
    __asm volatile("vmsr fpscr, %0" : : "r"(saved_fp) : "memory");
#endif
}
void test_div_f32_values(void) { div_values(); }
void test_div_f32_broadcast(void) { div_broadcast(); }
void test_div_f32_invalid(void) { div_invalid(); }
