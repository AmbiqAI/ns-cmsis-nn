/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal Zephyr smoke for the ns-cmsis-nn module.
 *
 * Goal: force the toolchain to actually resolve a CMSIS-NN symbol from
 * the ns-cmsis-nn module and emit a `west build` failure if the Zephyr
 * integration (`zephyr/CMakeLists.txt`, `zephyr/Kconfig`, the SSoT
 * include) breaks.
 *
 * The app does no real inference; it calls `arm_relu_q15` on a tiny
 * fixed buffer and prints the packed version macro.
 */

#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"

static int16_t smoke_buf[8] = { -3, -2, -1, 0, 1, 2, 3, 4 };

int main(void)
{
	arm_relu_q15(smoke_buf, ARRAY_SIZE(smoke_buf));

	printk("ns-cmsis-nn zephyr smoke OK, NS_CMSIS_NN_VERSION=%u, "
	       "buf[0]=%d buf[7]=%d\n",
	       (unsigned int)NS_CMSIS_NN_VERSION,
	       (int)smoke_buf[0], (int)smoke_buf[7]);

	return 0;
}
