/**
 * @file ns_ambiqsuite_harness_fvp.h
 * @brief FVP-compatible stub for AmbiqSuite harness functions
 *
 * This file provides minimal implementations of AmbiqSuite harness functions
 * for FVP (Fixed Virtual Platform) testing environment.
 */

#ifndef NS_AMBIQSUITE_HARNESS_FVP_H
#define NS_AMBIQSUITE_HARNESS_FVP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// FVP-compatible printf function
#define ns_lp_printf printf

// Mock timer functions
typedef struct
{
    uint32_t start_time;
    uint32_t duration;
} ns_timer_t;

static inline void ns_timer_init(ns_timer_t *timer)
{
    timer->start_time = 0;
    timer->duration = 0;
}

static inline void ns_timer_start(ns_timer_t *timer)
{
    timer->start_time = 0; // Mock implementation
}

// static inline uint32_t ns_timer_get_us(ns_timer_t *timer) {
//     return 1000; // Mock 1ms duration
// }

// Mock core functions
static inline void ns_core_init(void)
{
    // No-op for FVP
}

static inline void ns_core_deinit(void)
{
    // No-op for FVP
}

// Mock power management
static inline void ns_power_init(void)
{
    // No-op for FVP
}

static inline void ns_power_deinit(void)
{
    // No-op for FVP
}

// Mock memory management
static inline void *ns_malloc(size_t size) { return malloc(size); }

static inline void ns_free(void *ptr) { free(ptr); }

// // Mock GPIO functions (stubs)
// #define AM_HAL_GPIO_PINCFG_OUTPUT 0
// static inline void am_hal_gpio_pincfg_output(uint32_t pin) {
//     // No-op for FVP
// }

// Mock BSP functions (stubs)
static inline void am_bsp_low_power_init(void)
{
    // No-op for FVP
}

static inline void am_bsp_low_power_exit(void)
{
    // No-op for FVP
}

// // Mock utility functions
// static inline void am_util_delay_ms(uint32_t ms) {
//     // No-op for FVP
// }

// static inline void am_util_delay_us(uint32_t us) {
//     // No-op for FVP
// }

// Mock interrupt functions
static inline void am_hal_interrupt_master_enable(void)
{
    // No-op for FVP
}

static inline void am_hal_interrupt_master_disable(void)
{
    // No-op for FVP
}

// Mock system functions
static inline void am_hal_sysctrl_sleep(void)
{
    // No-op for FVP
}

static inline void am_hal_sysctrl_deepsleep(void)
{
    // No-op for FVP
}

// Mock TCM (Tightly Coupled Memory) macros for FVP
#define NS_PUT_IN_TCM
#define NS_PUT_IN_ITCM
#define NS_PUT_IN_DTCM

#ifdef __cplusplus
}
#endif

#endif // NS_AMBIQSUITE_HARNESS_FVP_H
