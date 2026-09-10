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

/* The image entry point.
 *
 * This has to be a FUNCTION, not the anchor array above. GNU ld and LLD
 * will happily take a data symbol for --entry, but armlink rejects it:
 *   L6204E: Entry point (0x...) does not point to an instruction.
 * All three toolchains therefore enter here, so the smoke link stays one
 * shape rather than forking per linker.
 *
 * It is never executed -- the smoke ELF is linked and inspected, never
 * run -- which is why calling through deliberately mismatched signatures
 * is harmless here. Referencing the array keeps it, and the kernels it
 * points at, live in the final image.
 */
void ns_cmsis_nn_smoke_entry(void);

void ns_cmsis_nn_smoke_entry(void)
{
    const size_t n = sizeof(ns_cmsis_nn_smoke_refs) / sizeof(ns_cmsis_nn_smoke_refs[0]);

    for (size_t i = 0U; i < n; ++i)
    {
        ns_cmsis_nn_smoke_refs[i]();
    }
}
