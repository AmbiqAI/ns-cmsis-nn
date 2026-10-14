// SPDX-FileCopyrightText: Copyright 2026 Ambiq
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#pragma once
#include <stdint.h>
#include <stddef.h>
#define ARM_MATH_MVEI 1
using arm_cmsis_nn_status = int;
constexpr int ARM_CMSIS_NN_SUCCESS = 0;
using mve_pred16_t = unsigned;
struct int8x16_t { int8_t v[16]{}; };
struct int32x4_t { int32_t v[4]{}; };
constexpr unsigned vctp8q(unsigned n) { return n < 16 ? n : 16; }
constexpr unsigned vctp32q(unsigned n) { return n < 4 ? n : 4; }
constexpr int8x16_t vldrbq_z_s8(const int8_t *p, unsigned n) {
    int8x16_t x{}; for(unsigned i=0;i<n;i++) x.v[i]=p[i]; return x;
}
constexpr int32_t vaddvaq_s8(int32_t a, int8x16_t x) {
    for(int i=0;i<16;i++) a+=x.v[i]; return a;
}
template<class T> constexpr void *probe_memcpy(T *p, const T *q, size_t n) {
    for(size_t i=0;i<n/sizeof(T);i++) p[i]=q[i]; return p;
}
template<class T> constexpr void *probe_memset(T *p, int value, size_t n) {
    if(value!=0) throw "only zero initialization is modeled";
    for(size_t i=0;i<n/sizeof(T);i++) p[i]=0; return p;
}
#define memcpy probe_memcpy
#define memset probe_memset
constexpr int32x4_t vldrwq_z_s32(const int32_t *p,unsigned n) {
    int32x4_t x{}; for(unsigned i=0;i<n;i++) x.v[i]=p[i]; return x;
}
// Identity scaling isolates pointer formation; target tests cover requantization.
constexpr int32x4_t arm_requantize_mve(int32x4_t x,int multiplier,int shift) {
    if(multiplier!=1 || shift!=0) throw "unsupported probe scaling"; return x;
}
constexpr int32x4_t vaddq_n_s32(int32x4_t x,int a) {
    for(int i=0;i<4;i++) x.v[i]+=a; return x;
}
constexpr int32x4_t vdupq_n_s32(int a) {
    int32x4_t x{}; for(int i=0;i<4;i++) x.v[i]=a; return x;
}
constexpr int32x4_t vmaxq_s32(int32x4_t x,int32x4_t y) {
    for(int i=0;i<4;i++) if(x.v[i]<y.v[i]) x.v[i]=y.v[i]; return x;
}
constexpr int32x4_t vminq_s32(int32x4_t x,int32x4_t y) {
    for(int i=0;i<4;i++) if(x.v[i]>y.v[i]) x.v[i]=y.v[i]; return x;
}
constexpr void vstrbq_p_s32(int8_t *p,int32x4_t x,unsigned n) {
    for(unsigned i=0;i<n;i++) p[i]=(int8_t)x.v[i];
}
