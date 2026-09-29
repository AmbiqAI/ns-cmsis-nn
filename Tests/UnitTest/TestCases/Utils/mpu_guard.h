/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * MPU guard gap for operand-bounds tests on the Corstone-300 test platform. An operand placed with guard_place() or
 * guard_end() ends exactly where GUARD_BLOCK unmapped bytes begin, so any access past it faults. Include once per test
 * executable: it defines HardFault_Handler.
 */
#pragma once

#if defined(ARM_MATH_MVEI) && defined(USING_FVP_CORSTONE_300)
    #include <inttypes.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #define MPU_GUARD_AVAILABLE 1
    #define GUARD_BLOCK 32
    #ifndef GUARD_OFFSET
        #define GUARD_OFFSET 2048
    #endif

_Static_assert(GUARD_OFFSET % GUARD_BLOCK == 0, "GUARD_OFFSET must be a multiple of GUARD_BLOCK");

/* Operands under test end at guard_arena + GUARD_OFFSET; the next GUARD_BLOCK bytes are left unmapped. */
static int8_t guard_arena[GUARD_OFFSET + GUARD_BLOCK] __attribute__((aligned(GUARD_BLOCK)));

/* The startup code leaves MemManage disabled, so an access to the gap escalates to HardFault. Report it and end the
   run rather than spin in the default handler. */
void HardFault_Handler(void)
{
    printf("HardFault: CFSR=0x%08" PRIx32 " MMFAR=0x%08" PRIx32 "\n", SCB->CFSR, SCB->MMFAR);
    exit(1);
}

/* Maps everything except the gap, with no default background map, so any access to the gap faults. RLAR limits are
   inclusive 32-byte granules: region 0 ends at gap - 1 and region 1 starts at gap + GUARD_BLOCK. */
static void guard_gap_enable(void)
{
    const uint32_t gap = (uint32_t)&guard_arena[GUARD_OFFSET];
    ARM_MPU_Disable();
    ARM_MPU_SetMemAttr(0, ARM_MPU_ATTR(ARM_MPU_ATTR_NON_CACHEABLE, ARM_MPU_ATTR_NON_CACHEABLE));
    ARM_MPU_SetRegion(0, ARM_MPU_RBAR(0, ARM_MPU_SH_NON, 0, 1, 0), ARM_MPU_RLAR(gap - GUARD_BLOCK, 0));
    ARM_MPU_SetRegion(1, ARM_MPU_RBAR(gap + GUARD_BLOCK, ARM_MPU_SH_NON, 0, 1, 0), ARM_MPU_RLAR(0xFFFFFFE0U, 0));
    MPU->CTRL = MPU_CTRL_ENABLE_Msk;
    __DSB();
    __ISB();
}

static void guard_gap_disable(void)
{
    MPU->CTRL = 0;
    __DSB();
    __ISB();
    ARM_MPU_ClrRegion(0);
    ARM_MPU_ClrRegion(1);
}

/* Storage of the given size that ends exactly at the gap. */
static void *guard_end(size_t bytes) { return &guard_arena[GUARD_OFFSET - bytes]; }

static void *guard_place(const void *src, size_t bytes) { return memcpy(guard_end(bytes), src, bytes); }
#endif
