/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Consumer side of the build-target check for a published archive: compiled with the archive's arch flags and the
 * ARM_NN_EXPECT_* values of its CPU (smoke_staticlib.sh passes them as -D), so the compile proves the guard accepts
 * that build and the link proves the archive defines that build's stamp.
 */

#include "arm_nn_build_target_check.h"

int ns_cmsis_nn_smoke_build_check(void);

int ns_cmsis_nn_smoke_build_check(void) { return ARM_NN_LIBRARY_STAMP() == 1U ? 0 : -1; }
