/* SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include <stddef.h>
#include <stdint.h>
#define ARM_NN_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARM_NN_MIN(a, b) ((a) < (b) ? (a) : (b))
#define ARM_MATH_AUTOVECTORIZE 1
using mve_pred16_t = unsigned;
struct int32x4_t
{
    int32_t v[4];
    constexpr int32_t &operator[](int i) { return v[i]; }
    constexpr int32_t operator[](int i) const { return v[i]; }
};
struct uint32x4_t
{
    uint32_t v[4];
    constexpr uint32_t operator[](int i) const { return v[i]; }
};
using float32_t = float;
using float16_t = float;
struct float32x4_t
{
    float v[4];
};
struct float16x8_t
{
    float v[8];
};
constexpr int32x4_t vdupq_n_s32(int x) { return {{x, x, x, x}}; }
constexpr mve_pred16_t vctp32q(unsigned n) { return n < 4 ? n : 4; }
constexpr mve_pred16_t vctp16q(unsigned n) { return n < 8 ? n : 8; }
constexpr int32x4_t vldrbq_z_s32(const int8_t *p, unsigned n)
{
    int32x4_t r = {};
    for (unsigned i = 0; i < n; i++)
        r[i] = p[i];
    return r;
}
constexpr int32x4_t vldrbq_s32(const int8_t *p) { return vldrbq_z_s32(p, 4); }
constexpr int32x4_t vldrwq_s32(const int32_t *p) { return {{p[0], p[1], p[2], p[3]}}; }
constexpr uint32x4_t vld1q_u32(const uint32_t *p) { return {{p[0], p[1], p[2], p[3]}}; }
constexpr int32x4_t vaddq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; i++)
        a[i] += b[i];
    return a;
}
constexpr int32x4_t vaddq_n_s32(int32x4_t a, int b) { return vaddq_s32(a, vdupq_n_s32(b)); }
constexpr int32x4_t vmulq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; i++)
        a[i] *= b[i];
    return a;
}
constexpr int32x4_t vmaxq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; i++)
        a[i] = ARM_NN_MAX(a[i], b[i]);
    return a;
}
constexpr int32x4_t vminq_s32(int32x4_t a, int32x4_t b)
{
    for (int i = 0; i < 4; i++)
        a[i] = ARM_NN_MIN(a[i], b[i]);
    return a;
}
constexpr void vstrbq_scatter_offset_s32(int8_t *p, uint32x4_t o, int32x4_t x)
{
    for (int i = 0; i < 4; i++)
        p[o[i]] = x[i];
}
constexpr void vstrbq_s32(int8_t *p, int32x4_t x)
{
    for (int i = 0; i < 4; i++)
        p[i] = x[i];
}
/* Requantization has no pointer state. Identity makes the access oracle a raw dot product. */
constexpr int32_t arm_nn_requantize(int32_t x, int32_t, int32_t) { return x; }
constexpr int32x4_t arm_requantize_mve(int32x4_t x, int32_t, int32_t) { return x; }
constexpr int32x4_t arm_requantize_mve_rshift(int32x4_t x, int32_t, int32_t) { return x; }
constexpr int32x4_t arm_requantize_mve_32x4(int32x4_t x, int32x4_t, int32x4_t) { return x; }
constexpr int32x4_t arm_requantize_mve_32x4_rshift(int32x4_t x, int32x4_t, int32x4_t) { return x; }
using arm_cmsis_nn_status = int;
constexpr int ARM_CMSIS_NN_SUCCESS = 0;
constexpr uint32x4_t vidupq_u32(unsigned n, int step) { return {{n, n + step, n + 2 * step, n + 3 * step}}; }
constexpr uint32x4_t vmulq(uint32x4_t a, unsigned b)
{
    for (int i = 0; i < 4; i++)
        a.v[i] *= b;
    return a;
}
constexpr float32x4_t vldrwq_gather_shifted_offset_z(const float *p, uint32x4_t o, unsigned n)
{
    float32x4_t r = {};
    for (unsigned i = 0; i < n; i++)
        r.v[i] = p[o[i]];
    return r;
}
constexpr float16x8_t vldrhq_gather_shifted_offset_z(const float *p, struct uint16x8_t o, unsigned n);
constexpr void vst1q_p(float *p, float32x4_t a, unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        p[i] = a.v[i];
}
struct uint16x8_t
{
    uint16_t v[8];
    constexpr uint16_t operator[](int i) const { return v[i]; }
};
constexpr uint16x8_t vidupq_u16(unsigned n, int step)
{
    uint16x8_t r = {};
    for (int i = 0; i < 8; i++)
        r.v[i] = n + i * step;
    return r;
}
constexpr uint16x8_t vmulq(uint16x8_t a, uint16_t b)
{
    for (int i = 0; i < 8; i++)
        a.v[i] *= b;
    return a;
}
constexpr float16x8_t vldrhq_gather_shifted_offset_z(const float *p, uint16x8_t o, unsigned n)
{
    float16x8_t r = {};
    for (unsigned i = 0; i < n; i++)
        r.v[i] = p[o[i]];
    return r;
}
constexpr void vst1q_p(float *p, float16x8_t a, unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        p[i] = a.v[i];
}
constexpr int32_t arm_nn_read_s8x4(const int8_t *p)
{
    uint32_t r = 0;
    for (int i = 0; i < 4; i++)
        r |= (uint32_t)(uint8_t)p[i] << (8 * i);
    return (int32_t)r;
}
constexpr int32_t arm_nn_read_s8x4_ia(const int8_t **p)
{
    int32_t r = arm_nn_read_s8x4(*p);
    *p += 4;
    return r;
}
constexpr int32_t SXTB16(int32_t x)
{
    return (uint16_t)(int16_t)(int8_t)x | ((uint32_t)(uint16_t)(int16_t)(int8_t)(x >> 16) << 16);
}
constexpr int32_t SXTB16_RORn(int32_t x, int n) { return SXTB16((uint32_t)x >> n | (uint32_t)x << (32 - n)); }
constexpr int32_t SMLAD(int32_t a, int32_t b, int32_t c)
{
    return c + (int16_t)a * (int16_t)b + (int16_t)(a >> 16) * (int16_t)(b >> 16);
}
