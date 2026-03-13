#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_reduce_sum_s16.c"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

void tearDown(void) {}

void test_reduce_sum_s16_axis_2_3(void) { reduce_sum_s16_axis_2_3(); }

void test_reduce_sum_s16(void) { reduce_sum_s16(); }
