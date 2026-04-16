#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_rsqrt_s16.c"
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

void tearDown(void)
{
}

void test_rsqrt_s16_universal_matches_reference(void)
{
    rsqrt_s16_universal_matches_reference();
}

void test_rsqrt_s16_per_op_matches_reference(void)
{
    rsqrt_s16_per_op_matches_reference();
}

void test_rsqrt_s16_per_op_applies_input_offset_before_lookup(void)
{
    rsqrt_s16_per_op_applies_input_offset_before_lookup();
}

void test_rsqrt_s16_universal_rescales_and_applies_input_offset(void)
{
    rsqrt_s16_universal_rescales_and_applies_input_offset();
}

void test_rsqrt_s16_rejects_negative_input(void)
{
    rsqrt_s16_rejects_negative_input();
}
