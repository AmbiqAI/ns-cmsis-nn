/* SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK */
#include <stdlib.h>

#include "../test_arm_int32_predicates.c"
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
void test_comparisons(void) { comparisons(); }
void test_selections(void) { selections(); }
void test_rejected_arguments(void) { rejected_arguments(); }
