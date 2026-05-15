/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Smoke-link source for the staticlib release pipeline.
 *
 * Goal: prove that the prebuilt cmsis-nn archive's symbols are present
 * and that the archive is mechanically linkable, NOT that the kernels
 * run correctly. We don't run anything — we just produce an ELF and
 * inspect it.
 *
 * We resolve one symbol from each kernel group taxonomically guaranteed
 * to be in the archive (per scripts/ns_cmsis_nn.cmake). If any of them
 * is missing the linker fails the smoke job.
 *
 * No CMSIS Core / device headers are needed: we only reference the
 * symbols by name, never call them.
 */

#include <stddef.h>
#include <stdint.h>

/* Forward-declare one routine from each major group. The exact
 * signatures don't matter to the linker — only that the symbol
 * resolves. */
extern void arm_relu_q7(int8_t *data, uint16_t size);
extern void arm_softmax_s8(const int8_t *input, const int32_t num_rows,
                           const int32_t row_size, const int32_t mult,
                           const int32_t shift, const int32_t diff_min,
                           int8_t *output);
extern int  arm_convolve_s8(void);
extern int  arm_fully_connected_s8(void);
extern int  arm_max_pool_s8(void);
extern int  arm_avgpool_s8(void);
extern int  arm_elementwise_add_s8(void);
extern int  arm_elementwise_mul_s8(void);

/* Single anchor that references the symbols so the linker keeps them
 * in the final ELF. */
void (* const ns_cmsis_nn_smoke_refs[])(void) = {
    (void (*)(void)) arm_relu_q7,
    (void (*)(void)) arm_softmax_s8,
    (void (*)(void)) arm_convolve_s8,
    (void (*)(void)) arm_fully_connected_s8,
    (void (*)(void)) arm_max_pool_s8,
    (void (*)(void)) arm_avgpool_s8,
    (void (*)(void)) arm_elementwise_add_s8,
    (void (*)(void)) arm_elementwise_mul_s8,
};
