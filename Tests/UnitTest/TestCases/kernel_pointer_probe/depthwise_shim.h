/* SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include "pointer_shim.h"
constexpr int32x4_t vldrhq_z_s32(const int16_t *p, unsigned n)
{
    int32x4_t r = {};
    for (unsigned i = 0; i < n; i++)
        r[i] = p[i];
    return r;
}
constexpr int32x4_t vldrhq_s32(const int16_t *p) { return vldrhq_z_s32(p, 4); }
constexpr int32x4_t vldrhq_gather_shifted_offset_z_s32(const int16_t *p, uint32x4_t o, unsigned n)
{
    int32x4_t r = {};
    for (unsigned i = 0; i < n; i++)
        r[i] = p[o[i]];
    return r;
}
constexpr int32x4_t vldrhq_gather_shifted_offset_s32(const int16_t *p, uint32x4_t o)
{
    return vldrhq_gather_shifted_offset_z_s32(p, o, 4);
}
constexpr int32x4_t vmlaq_n_s32(int32x4_t a, int32x4_t b, int c) { return vaddq_s32(a, vmulq_s32(b, vdupq_n_s32(c))); }
constexpr void vstrhq_p_s32(int16_t *p, int32x4_t x, unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        p[i] = x[i];
}
constexpr void vstrhq_s32(int16_t *p, int32x4_t x) { vstrhq_p_s32(p, x, 4); }
#define DW_S16_IN_VEC 0
#define DW_S16_IN_DUP 1
#define DW_S16_IN_GATHER 2
#define DW_S16_IN_VEC_P 3
#define DW_S16_IN_GATHER_P 4
struct dw_s16_rq
{
    int32x4_t bias;
};
constexpr dw_s16_rq dw_s16_rq_load(const int32_t *, const int32_t *, const int64_t *b, unsigned n)
{
    dw_s16_rq r = {};
    for (unsigned i = 0; i < n; i++)
        r.bias[i] = b ? b[i] : 0;
    return r;
}
constexpr int32x4_t dw_s16_rq_apply(int32x4_t a, const dw_s16_rq *, int lo, int hi)
{
    return vminq_s32(vmaxq_s32(a, vdupq_n_s32(lo)), vdupq_n_s32(hi));
}
constexpr int32x4_t
dw_s16_out(int32x4_t a, const dw_s16_rq *r, int, const int64_t *, const int32_t *, const int32_t *, int, int lo, int hi)
{
    return dw_s16_rq_apply(a, r, lo, hi);
}

/* Model integer addresses as an object origin plus an unsigned byte displacement. Only conversion back to a
 * pointer forms a C++ pointer, so constant evaluation checks the real load/store addresses. The one-instruction
 * address step is modeled as addition here; target execution checks its assembly implementation separately. */
struct probe_address
{
    const int8_t *bytes = nullptr;
    const int16_t *halves = nullptr;
    int16_t *writable = nullptr;
    uint32_t offset = 0;
    constexpr probe_address(uint32_t n) : offset(n) {}
    constexpr probe_address(const int8_t *p) : bytes(p) {}
    constexpr probe_address(const int16_t *p) : halves(p) {}
    constexpr probe_address(int16_t *p) : halves(p), writable(p) {}
    constexpr probe_address operator+(probe_address n) const
    {
        if (n.bytes || n.halves) throw "address added to address";
        probe_address result = *this;
        result.offset += n.offset;
        return result;
    }
    constexpr probe_address &operator+=(probe_address n) { return *this = *this + n; }
    constexpr probe_address operator*(uint32_t n) const
    {
        if (bytes || halves) throw "address multiplied";
        return probe_address(offset * n);
    }
    constexpr operator const int8_t *() const
    {
        if (!bytes) throw "wrong address type";
        return bytes + offset;
    }
    constexpr operator const int16_t *() const
    {
        if (!halves || offset % 2) throw "wrong address type or alignment";
        return halves + offset / 2;
    }
    constexpr operator int16_t *() const
    {
        if (!writable || offset % 2) throw "wrong address type or alignment";
        return writable + offset / 2;
    }
};
#define uintptr_t probe_address
constexpr probe_address dw_s16_step(probe_address address, probe_address bytes) { return address + bytes; }
