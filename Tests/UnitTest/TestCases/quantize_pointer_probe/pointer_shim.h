// SPDX-FileCopyrightText: Copyright 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#define ARM_MATH_MVEF 1
using mve_pred16_t = unsigned;
struct float32x4_t { float v[4]; };
struct int32x4_t { int32_t v[4]; };
enum arm_cmsis_nn_status { ARM_CMSIS_NN_SUCCESS, ARM_CMSIS_NN_ARG_ERROR };
constexpr mve_pred16_t vctp32q(unsigned n) { return n < 4 ? n : 4; }
constexpr float32x4_t vldrwq_z_f32(const float *p, unsigned n) {
    float32x4_t r{}; for (unsigned i=0;i<n;i++) r.v[i]=p[i]; return r;
}
constexpr int32x4_t vdupq_n_s32(int32_t n) { return {{n,n,n,n}}; }
constexpr float32x4_t vmulq_n_f32(float32x4_t a, float n) {
    for (int i=0;i<4;i++) a.v[i]*=n; return a;
}
// Only exact small integers enter this probe; target numerical tests cover rounding/saturation.
constexpr int32x4_t vcvtaq_s32_f32(float32x4_t a) {
    int32x4_t r{}; for (int i=0;i<4;i++) r.v[i]=(int32_t)a.v[i]; return r;
}
constexpr int32x4_t vmaxq_s32(int32x4_t a, int32x4_t b) {
    for (int i=0;i<4;i++) if (a.v[i]<b.v[i]) a.v[i]=b.v[i]; return a;
}
constexpr int32x4_t vminq_s32(int32x4_t a, int32x4_t b) {
    for (int i=0;i<4;i++) if (a.v[i]>b.v[i]) a.v[i]=b.v[i]; return a;
}
constexpr int32x4_t vaddq_n_s32(int32x4_t a, int n) {
    for (int i=0;i<4;i++) a.v[i]+=n; return a;
}
constexpr void vstrbq_p_s32(int8_t *p, int32x4_t a, unsigned n) {
    for (unsigned i=0;i<n;i++) p[i]=(int8_t)a.v[i];
}
constexpr void vstrhq_p_s32(int16_t *p, int32x4_t a, unsigned n) {
    for (unsigned i=0;i<n;i++) p[i]=(int16_t)a.v[i];
}
