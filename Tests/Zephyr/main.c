/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include <arm_nnfunctions.h>
#include <string.h>
#include <zephyr/kernel.h>

BUILD_ASSERT(IS_ENABLED(CONFIG_ENFORCE_ZEPHYR_STDINT));
BUILD_ASSERT(IS_ENABLED(CONFIG_ARMV8_1_M_MVEF));
BUILD_ASSERT(!IS_ENABLED(CONFIG_NS_CMSIS_NN_USE_PREBUILT));

int main(void)
{
    const cmsis_nn_dims shape = {.n = 1, .h = 1, .w = 2, .c = 3};
    const cmsis_nn_dims axis = {.w = 1};
    const cmsis_nn_dims row = {.n = 1, .h = 1, .w = 1, .c = 3};
    // Reduce across rows. Keep the first signed zero, the negative subnormal and the canonical NaN.
    const uint32_t bits[] = {0x80000000, 0, 0x3f800000, 0, 0x80000001, 0x7fc01234};
    const uint32_t expected[] = {0x80000000, 0x80000001, 0x7fc00000, 0x3f800000};
    float32_t input[6], output[4] = {0, 0, 0, 1};
    memcpy(input, bits, sizeof(input));
    const float16_t a[] = {1, 2, 3};
    float16_t sum[3];

    if (arm_reduce_min_f32(input, &shape, &axis, output, &row) != ARM_CMSIS_NN_SUCCESS ||
        memcmp(output, expected, sizeof(expected)) != 0 ||
        arm_elementwise_add_f16(a, a, sum, -10, 10, 3) != ARM_CMSIS_NN_SUCCESS || sum[0] != 2 || sum[1] != 4 ||
        sum[2] != 6)
    {
        printk("heliaCORE Zephyr smoke FAIL\n");
        return 1;
    }
    printk("heliaCORE Zephyr smoke PASS\n");
    return 0;
}
