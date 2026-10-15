/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

/* Constant-evaluable stand-ins for the MVE intrinsics the probed kernels use, for a host C++ compiler. Lanes follow
 * the MVE definitions; predicated loads and stores touch active lanes only, as on the hardware. */
#ifndef KERNEL_POINTER_PROBE_MVE_SHIM_H
#define KERNEL_POINTER_PROBE_MVE_SHIM_H

#include <stdint.h>

struct int16x8_t
{
    int16_t v[8];
};
struct int32x4_t
{
    int32_t v[4];
};
struct uint32x4_t
{
    uint32_t v[4];
};
typedef uint16_t mve_pred16_t;

constexpr int32_t shim_sat(int64_t x, int64_t lo, int64_t hi) { return (int32_t)(x < lo ? lo : (x > hi ? hi : x)); }
constexpr int32_t shim_wrap(int64_t x) { return (int32_t)(uint32_t)(uint64_t)x; }
constexpr bool shim_on(mve_pred16_t p, int lane, int bytes) { return (p >> (lane * bytes)) & 1; }
constexpr mve_pred16_t shim_ctp(uint32_t n, uint32_t lanes, int bytes)
{
    const uint32_t k = n < lanes ? n : lanes;
    return (mve_pred16_t)(k * bytes >= 16 ? 0xFFFF : (1u << (k * bytes)) - 1);
}
constexpr mve_pred16_t vctp16q(uint32_t n) { return shim_ctp(n, 8, 2); }
constexpr mve_pred16_t vctp32q(uint32_t n) { return shim_ctp(n, 4, 4); }
constexpr int32_t shim_shl(int32_t x, int32_t s)
{
    if (s >= 32)
        return 0;
    if (s >= 0)
        return shim_wrap((int64_t)((uint64_t)(uint32_t)x << s));
    return s <= -32 ? (x < 0 ? -1 : 0) : x >> -s;
}
constexpr int32_t shim_rshl(int32_t x, int32_t s)
{
    if (s >= 0)
        return shim_shl(x, s);
    if (s <= -32)
        return 0;
    return (int32_t)(((int64_t)x + ((int64_t)1 << (-s - 1))) >> -s);
}
constexpr int32_t shim_qrdmulh(int32_t a, int32_t b)
{
    return shim_sat(((int64_t)a * b + ((int64_t)1 << 30)) >> 31, INT32_MIN, INT32_MAX);
}

constexpr int16x8_t vdupq_n_s16(int16_t a) { return {{a, a, a, a, a, a, a, a}}; }
constexpr int32x4_t vdupq_n_s32(int32_t a) { return {{a, a, a, a}}; }
constexpr int16x8_t vldrbq_z_s16(const int8_t *base, mve_pred16_t p)
{
    int16x8_t r = {};
    for (int i = 0; i < 8; ++i)
        r.v[i] = shim_on(p, i, 2) ? base[i] : 0;
    return r;
}
constexpr int32x4_t vldrbq_z_s32(const int8_t *base, mve_pred16_t p)
{
    int32x4_t r = {};
    for (int i = 0; i < 4; ++i)
        r.v[i] = shim_on(p, i, 4) ? base[i] : 0;
    return r;
}
constexpr void vstrbq_p_s16(int8_t *base, int16x8_t a, mve_pred16_t p)
{
    for (int i = 0; i < 8; ++i)
        if (shim_on(p, i, 2))
            base[i] = (int8_t)a.v[i];
}
constexpr void vstrbq_p_s32(int8_t *base, int32x4_t a, mve_pred16_t p)
{
    for (int i = 0; i < 4; ++i)
        if (shim_on(p, i, 4))
            base[i] = (int8_t)a.v[i];
}
constexpr int16x8_t vaddq_n_s16(int16x8_t a, int16_t b)
{
    for (int i = 0; i < 8; ++i)
        a.v[i] = (int16_t)(uint16_t)(a.v[i] + b);
    return a;
}
constexpr int16x8_t vqaddq_n_s16(int16x8_t a, int16_t b)
{
    for (int i = 0; i < 8; ++i)
        a.v[i] = (int16_t)shim_sat(a.v[i] + b, INT16_MIN, INT16_MAX);
    return a;
}
constexpr int16x8_t vmaxq_s16(int16x8_t a, int16x8_t b)
{
    for (int i = 0; i < 8; ++i)
        a.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i];
    return a;
}
constexpr int16x8_t vminq_s16(int16x8_t a, int16x8_t b)
{
    for (int i = 0; i < 8; ++i)
        a.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i];
    return a;
}
constexpr int32x4_t vmullbq_int_s16(int16x8_t a, int16x8_t b)
{
    int32x4_t r = {};
    for (int i = 0; i < 4; ++i)
        r.v[i] = a.v[2 * i] * b.v[2 * i];
    return r;
}
constexpr int32x4_t vmulltq_int_s16(int16x8_t a, int16x8_t b)
{
    int32x4_t r = {};
    for (int i = 0; i < 4; ++i)
        r.v[i] = a.v[2 * i + 1] * b.v[2 * i + 1];
    return r;
}
constexpr int16x8_t vqmovnbq_s32(int16x8_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[2 * i] = (int16_t)shim_sat(b.v[i], INT16_MIN, INT16_MAX);
    return a;
}
constexpr int16x8_t vqmovntq_s32(int16x8_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[2 * i + 1] = (int16_t)shim_sat(b.v[i], INT16_MIN, INT16_MAX);
    return a;
}
constexpr int32x4_t vaddq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = shim_wrap((int64_t)a.v[i] + b.v[i]);
    return a;
}
constexpr int32x4_t vaddq_n_s32(int32x4_t a, int32_t b) { return vaddq_s32(a, vdupq_n_s32(b)); }
constexpr int32x4_t vqaddq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = shim_sat((int64_t)a.v[i] + b.v[i], INT32_MIN, INT32_MAX);
    return a;
}
constexpr int32x4_t vmaxq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i];
    return a;
}
constexpr int32x4_t vminq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i];
    return a;
}
constexpr int32x4_t vshlq_s32(int32x4_t a, int32x4_t s)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = shim_shl(a.v[i], s.v[i]);
    return a;
}
constexpr int32x4_t vshlq_r_s32(int32x4_t a, int32_t s) { return vshlq_s32(a, vdupq_n_s32(s)); }
constexpr int32x4_t vshrq_n_s32(int32x4_t a, int n) { return vshlq_s32(a, vdupq_n_s32(-n)); }
constexpr int32x4_t vrshlq_s32(int32x4_t a, int32x4_t s)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = shim_rshl(a.v[i], s.v[i]);
    return a;
}
constexpr int32x4_t vandq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = a.v[i] & b.v[i];
    return a;
}
constexpr int32x4_t vqrdmulhq_n_s32(int32x4_t a, int32_t b)
{
    for (int i = 0; i < 4; ++i)
        a.v[i] = shim_qrdmulh(a.v[i], b);
    return a;
}

#endif
