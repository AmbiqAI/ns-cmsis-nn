/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The differential cases: arm_gather_f32, and arm_gather_nd_f32 and, where binary16 is built,
 * arm_gather_f16 and arm_gather_nd_f16 (all four share Include/Internal/arm_nn_gather_common.h),
 * against the bodies they had (gather_previous.h).
 *
 * Each case first holds the library's prepare step (arm_nn_gather_prepare or
 * arm_nn_gather_nd_prepare) to the previous one on the same arguments: the statuses, and for an
 * accepted shape the output count and, where it is nonzero, every field of the plan. A shape both
 * accept is then run through both entries when every element the plan reads and writes lies in
 * the buffers (a huge gathered axis with small indices runs; a huge output does not, and is left
 * to the prepare comparison, which gd_planned_only counts). A run differs when the statuses
 * differ, when any byte of the two outputs differs (guards included), when a guard of the
 * entry's output moved, or when the input or the indices changed. Every buffer is prepared and
 * compared over what the case can touch and GD_GUARD elements either side.
 *
 * The count step's own decision (arm_nn_gather_count against gp_gather_count) is swept apart from
 * any shape, at element sizes of one to eight bytes. */

#include "../../../../Include/Internal/arm_nn_gather_common.h"
#include "gather_previous.h"
#include <stdio.h>
#include <string.h>
#include <unity.h>

#define GD_IN 4096
#define GD_OUT 4096
#define GD_IDX 512
#define GD_GUARD 16
#define GD_BYTE 0xa5
#define GD_REPORTED 8

static uint32_t gd_bits[GD_IN];
static float32_t gd_in32[GD_IN + 2 * GD_GUARD];
static float32_t gd_new32[GD_OUT + 2 * GD_GUARD];
static float32_t gd_old32[GD_OUT + 2 * GD_GUARD];
#if ARM_NN_ENABLE_F16
static float16_t gd_in16[GD_IN + 2 * GD_GUARD];
static float16_t gd_new16[GD_OUT + 2 * GD_GUARD];
static float16_t gd_old16[GD_OUT + 2 * GD_GUARD];
#endif
static int32_t gd_indices[GD_IDX + 2 * GD_GUARD];
static int32_t gd_index_copy[GD_IDX];
static unsigned long gd_cases;
static unsigned long gd_differences;
static unsigned long gd_planned_only;
static unsigned long gd_statuses[2];
static unsigned long gd_inner_runs[3];
static uint64_t gd_last_inner;
static char gd_message[224];

/* Kinds: GATHER binary32, GATHER_ND binary32, GATHER binary16, GATHER_ND binary16. */
#if ARM_NN_ENABLE_F16
    #define GD_KINDS 4
#else
    #define GD_KINDS 2
#endif
static const char *const gd_names[4] = {"gather_f32", "gather_nd_f32", "gather_f16", "gather_nd_f16"};

static int gd_nd(int kind) { return kind & 1; }

static size_t gd_width(int kind) { return kind < 2 ? sizeof(float32_t) : sizeof(uint16_t); }

/* One case's arguments. rank, irank, axis, batch are the parameter fields as passed (irank is
 * coords_rank for GATHER and indices_rank for GATHER_ND); the dims are passed as given, a NULL
 * where pass says so. */
typedef struct
{
    int kind;
    int32_t rank, irank, axis, batch;
    cmsis_nn_dims in, idx, out;
    int32_t index_count;
    long long out_count;
    int pass_input, pass_indices, pass_params, pass_output, pass_in_dims, pass_idx_dims, pass_out_dims;
} gd_case;

static unsigned char *gd_input_bytes(int kind)
{
#if ARM_NN_ENABLE_F16
    if (kind >= 2)
        return (unsigned char *)gd_in16;
#endif
    (void)kind;
    return (unsigned char *)gd_in32;
}

static unsigned char *gd_output_bytes(int kind, int fresh)
{
#if ARM_NN_ENABLE_F16
    if (kind >= 2)
        return (unsigned char *)(fresh ? gd_new16 : gd_old16);
#endif
    (void)kind;
    return (unsigned char *)(fresh ? gd_new32 : gd_old32);
}

static arm_cmsis_nn_status gd_entry(const gd_case *c, int fresh, const void *input, void *output)
{
    const cmsis_nn_gather_params gp = {c->axis, c->batch, c->rank, c->irank};
    const cmsis_nn_gather_nd_params np = {c->rank, c->irank, c->batch};
    const cmsis_nn_dims *in = c->pass_in_dims ? &c->in : NULL;
    const cmsis_nn_dims *idx = c->pass_idx_dims ? &c->idx : NULL;
    const cmsis_nn_dims *out = c->pass_out_dims ? &c->out : NULL;
    const int32_t *indices = c->pass_indices ? gd_indices + GD_GUARD : NULL;
    switch (c->kind)
    {
    case 0:
        return fresh ? arm_gather_f32(input, in, indices, idx, c->pass_params ? &gp : NULL, output, out)
                     : gp_gather_f32(input, in, indices, idx, c->pass_params ? &gp : NULL, output, out);
    case 1:
        return fresh ? arm_gather_nd_f32(input, in, indices, idx, c->pass_params ? &np : NULL, output, out)
                     : gp_gather_nd_f32(input, in, indices, idx, c->pass_params ? &np : NULL, output, out);
#if ARM_NN_ENABLE_F16
    case 2:
        return fresh ? arm_gather_f16(input, in, indices, idx, c->pass_params ? &gp : NULL, output, out)
                     : gp_gather_f16(input, in, indices, idx, c->pass_params ? &gp : NULL, output, out);
    case 3:
        return fresh ? arm_gather_nd_f16(input, in, indices, idx, c->pass_params ? &np : NULL, output, out)
                     : gp_gather_nd_f16(input, in, indices, idx, c->pass_params ? &np : NULL, output, out);
#endif
    default:
        return ARM_CMSIS_NN_ARG_ERROR;
    }
}

/* The prepare steps, library and previous, on the case's arguments, with the real index buffer
 * and the input and output buffers' addresses (a prepare step reads only the indices). Returns
 * the number of input elements the plan reads up to (one past the last), the output elements
 * it writes, or -1 where the two steps disagree. */
static int gd_prepare(const gd_case *c, arm_cmsis_nn_status *status, uint64_t *reads, uint64_t *writes)
{
    const cmsis_nn_gather_params gp = {c->axis, c->batch, c->rank, c->irank};
    const cmsis_nn_gather_nd_params np = {c->rank, c->irank, c->batch};
    const cmsis_nn_dims *in = c->pass_in_dims ? &c->in : NULL;
    const cmsis_nn_dims *idx = c->pass_idx_dims ? &c->idx : NULL;
    const cmsis_nn_dims *out = c->pass_out_dims ? &c->out : NULL;
    const int32_t *indices = c->pass_indices ? gd_indices + GD_GUARD : NULL;
    const void *input = c->pass_input ? (const void *)(gd_input_bytes(c->kind) + GD_GUARD * gd_width(c->kind)) : NULL;
    void *output = c->pass_output ? (void *)(gd_output_bytes(c->kind, 1) + GD_GUARD * gd_width(c->kind)) : NULL;
    const size_t width = gd_width(c->kind);
    *reads = 0;
    *writes = 0;
    if (!gd_nd(c->kind))
    {
        arm_nn_gather_plan fresh;
        gp_gather_plan previous;
        memset(&fresh, 0, sizeof(fresh));
        memset(&previous, 0, sizeof(previous));
        const arm_cmsis_nn_status got =
            arm_nn_gather_prepare(input, in, indices, idx, c->pass_params ? &gp : NULL, output, out, width, &fresh);
        const arm_cmsis_nn_status want =
            gp_gather_prepare(input, in, indices, idx, c->pass_params ? &gp : NULL, output, out, width, &previous);
        *status = want;
        if (got != want)
            return -1;
        if (want != ARM_CMSIS_NN_SUCCESS)
            return 0;
        if (fresh.output_count != previous.output_count)
            return -1;
        if (previous.output_count == 0)
            return 0;
        if (fresh.batches != previous.batches || fresh.outer != previous.outer || fresh.inner != previous.inner ||
            fresh.coords != previous.coords || fresh.axis_size != previous.axis_size)
            return -1;
        *writes = previous.output_count;
        gd_last_inner = previous.inner;
        if (*writes > GD_OUT)
            return 0;
        for (size_t b = 0; b < previous.batches; ++b)
            for (size_t o = 0; o < previous.outer; ++o)
                for (size_t k = 0; k < previous.coords; ++k)
                {
                    const uint64_t base =
                        ((uint64_t)b * previous.outer + o) * (uint64_t)previous.axis_size * previous.inner;
                    const uint64_t end =
                        base + (uint64_t)indices[b * previous.coords + k] * previous.inner + previous.inner;
                    *reads = end > *reads ? end : *reads;
                }
        return 0;
    }
    arm_nn_gather_nd_plan fresh;
    gp_gather_nd_plan previous;
    memset(&fresh, 0, sizeof(fresh));
    memset(&previous, 0, sizeof(previous));
    const arm_cmsis_nn_status got =
        arm_nn_gather_nd_prepare(input, in, indices, idx, c->pass_params ? &np : NULL, output, out, width, &fresh);
    const arm_cmsis_nn_status want =
        gp_gather_nd_prepare(input, in, indices, idx, c->pass_params ? &np : NULL, output, out, width, &previous);
    *status = want;
    if (got != want)
        return -1;
    if (want != ARM_CMSIS_NN_SUCCESS)
        return 0;
    if (fresh.output_count != previous.output_count)
        return -1;
    if (previous.output_count == 0)
        return 0;
    if (fresh.batches != previous.batches || fresh.slices != previous.slices || fresh.inner != previous.inner ||
        fresh.batch_stride != previous.batch_stride || fresh.width != previous.width ||
        fresh.batch_dims != previous.batch_dims)
        return -1;
    for (int d = 0; d < c->rank && d < 4; ++d)
        if (fresh.strides[d] != previous.strides[d])
            return -1;
    *writes = previous.output_count;
    gd_last_inner = previous.inner;
    if (*writes > GD_OUT)
        return 0;
    const int32_t *tuple = indices;
    for (size_t b = 0; b < previous.batches; ++b)
        for (size_t s = 0; s < previous.slices; ++s)
        {
            uint64_t offset = (uint64_t)b * previous.batch_stride;
            for (int32_t d = 0; d < previous.width; ++d)
                offset += (uint64_t)*tuple++ * previous.strides[previous.batch_dims + d];
            const uint64_t end = offset + previous.inner;
            *reads = end > *reads ? end : *reads;
        }
    return 0;
}

static void gd_differ(const char *pattern, const gd_case *c, const char *what, int got, int want)
{
    if (gd_differences < GD_REPORTED)
    {
        snprintf(gd_message,
                 sizeof(gd_message),
                 "difference: %s %s rank %ld irank %ld axis %ld batch %ld in {%ld, %ld, %ld, %ld} idx {%ld, %ld, %ld, "
                 "%ld} %s %d against %d",
                 gd_names[c->kind],
                 pattern,
                 (long)c->rank,
                 (long)c->irank,
                 (long)c->axis,
                 (long)c->batch,
                 (long)c->in.n,
                 (long)c->in.h,
                 (long)c->in.w,
                 (long)c->in.c,
                 (long)c->idx.n,
                 (long)c->idx.h,
                 (long)c->idx.w,
                 (long)c->idx.c,
                 what,
                 got,
                 want);
        TEST_MESSAGE(gd_message);
    }
    ++gd_differences;
}

/* One case on gd_bits (the input's encodings, the low half of each word for binary16) and the
 * index_count indices already in gd_indices. */
static void gd_run(const char *pattern, const gd_case *c)
{
    const size_t width = gd_width(c->kind);
    for (int32_t i = 0; i < GD_GUARD; ++i)
    {
        gd_indices[i] = INT32_C(0x5a5a5a5a);
        gd_indices[GD_GUARD + c->index_count + i] = INT32_C(0x5a5a5a5a);
    }
    memcpy(gd_index_copy, gd_indices + GD_GUARD, (size_t)c->index_count * sizeof(int32_t));
    arm_cmsis_nn_status status;
    uint64_t reads, writes;
    ++gd_cases;
    gd_last_inner = 0;
    if (gd_prepare(c, &status, &reads, &writes) < 0)
    {
        gd_differ(pattern, c, "prepare", 0, 0);
        return;
    }
    if (status == ARM_CMSIS_NN_SUCCESS && (writes > GD_OUT || reads > GD_IN))
    {
        --gd_cases;
        ++gd_planned_only;
        return;
    }
    ++gd_statuses[status == ARM_CMSIS_NN_SUCCESS];
    if (writes > 0)
        ++gd_inner_runs[gd_last_inner < 4 ? 0 : gd_last_inner == 4 ? 1 : 2];
    const size_t in_span = (size_t)reads;
    const size_t out_span = (size_t)writes;
    unsigned char *input = gd_input_bytes(c->kind);
    unsigned char *fresh = gd_output_bytes(c->kind, 1);
    unsigned char *previous = gd_output_bytes(c->kind, 0);
    memset(input, GD_BYTE, (in_span + 2 * GD_GUARD) * width);
    for (size_t i = 0; i < in_span; ++i)
        memcpy(input + (GD_GUARD + i) * width, &gd_bits[i], width);
    memset(fresh, GD_BYTE, (out_span + 2 * GD_GUARD) * width);
    memset(previous, GD_BYTE, (out_span + 2 * GD_GUARD) * width);
    const void *in_arg = c->pass_input ? (const void *)(input + GD_GUARD * width) : NULL;
    const arm_cmsis_nn_status got = gd_entry(c, 1, in_arg, c->pass_output ? fresh + GD_GUARD * width : NULL);
    const arm_cmsis_nn_status want = gd_entry(c, 0, in_arg, c->pass_output ? previous + GD_GUARD * width : NULL);
    if (got != want)
    {
        gd_differ(pattern, c, "status", (int)got, (int)want);
        return;
    }
    int differs = memcmp(fresh, previous, (out_span + 2 * GD_GUARD) * width) != 0;
    const size_t written = got == ARM_CMSIS_NN_SUCCESS ? out_span : 0;
    for (size_t i = 0; i < (out_span + 2 * GD_GUARD) * width; ++i)
    {
        const int inside = i >= GD_GUARD * width && i < (GD_GUARD + written) * width;
        if (!inside && fresh[i] != GD_BYTE)
            differs = 1;
    }
    for (size_t i = 0; i < (in_span + 2 * GD_GUARD) * width; ++i)
    {
        const int inside = i >= GD_GUARD * width && i < (GD_GUARD + in_span) * width;
        if (!inside && input[i] != GD_BYTE)
            differs = 1;
    }
    for (size_t i = 0; i < in_span; ++i)
        if (memcmp(input + (GD_GUARD + i) * width, &gd_bits[i], width) != 0)
            differs = 1;
    differs |= memcmp(gd_index_copy, gd_indices + GD_GUARD, (size_t)c->index_count * sizeof(int32_t)) != 0;
    for (int32_t i = 0; i < GD_GUARD; ++i)
        differs |=
            gd_indices[i] != INT32_C(0x5a5a5a5a) || gd_indices[GD_GUARD + c->index_count + i] != INT32_C(0x5a5a5a5a);
    if (differs)
        gd_differ(pattern, c, "bytes", (int)got, (int)want);
}

static uint32_t gd_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

/* Input patterns: random encodings (every class, NaN payloads of both signs included), and
 * specials only: both zeros, both infinities, quiet and signalling NaNs with the smallest and
 * largest payloads, the smallest subnormals. A copy must keep every bit of each. */
static const uint32_t gd_special32[] = {0,
                                        UINT32_C(0x80000000),
                                        UINT32_C(0x7f800000),
                                        UINT32_C(0xff800000),
                                        UINT32_C(0x7fc00000),
                                        UINT32_C(0xffc00001),
                                        UINT32_C(0x7f800001),
                                        UINT32_C(0xffbfffff),
                                        UINT32_C(0x00000001),
                                        UINT32_C(0x80000001)};
static const uint32_t gd_special16[] = {0,
                                        UINT32_C(0x8000),
                                        UINT32_C(0x7c00),
                                        UINT32_C(0xfc00),
                                        UINT32_C(0x7e00),
                                        UINT32_C(0xfe01),
                                        UINT32_C(0x7c01),
                                        UINT32_C(0xfdff),
                                        UINT32_C(0x0001),
                                        UINT32_C(0x8001)};
#define GD_SPECIALS ((int)(sizeof(gd_special32) / sizeof(gd_special32[0])))

static void gd_fill(int pattern, size_t width, uint32_t *state)
{
    const uint32_t *special = width == sizeof(float32_t) ? gd_special32 : gd_special16;
    const uint32_t mask = width == sizeof(float32_t) ? UINT32_MAX : UINT32_C(0xffff);
    for (int i = 0; i < GD_IN; ++i)
    {
        const uint32_t r = gd_next(state);
        gd_bits[i] = pattern == 0 ? (r & mask) : special[r % GD_SPECIALS];
    }
}

static cmsis_nn_dims gd_dims(const int32_t s[4], int32_t rank, int32_t rest)
{
    cmsis_nn_dims d = {rest, rest, rest, rest};
    if (rank > 0)
        d.n = s[0];
    if (rank > 1)
        d.h = s[1];
    if (rank > 2)
        d.w = s[2];
    if (rank > 3)
        d.c = s[3];
    return d;
}

static long long gd_product(const int32_t *s, int32_t from, int32_t to)
{
    long long p = 1;
    for (int32_t d = from; d < to; ++d)
        p *= s[d];
    return p;
}

/* A valid case of `kind` from shape (rank extents) and ishape (irank extents), the axis and
 * batch counts already normalised; the parameter fields take the negative forms where
 * `negative` says so. Returns 0 where the output rank passes 4. */
static int gd_build(gd_case *c,
                    int kind,
                    int32_t rank,
                    int32_t irank,
                    int32_t axis,
                    int32_t batch,
                    const int32_t shape[4],
                    const int32_t ishape[4],
                    int negative,
                    int32_t rest)
{
    int32_t oshape[4] = {1, 1, 1, 1};
    int32_t orank = 0;
    if (!gd_nd(kind))
    {
        if (rank + irank - batch - 1 > 4)
            return 0;
        for (int32_t d = 0; d < axis; ++d)
            oshape[orank++] = shape[d];
        for (int32_t d = batch; d < irank; ++d)
            oshape[orank++] = ishape[d];
        for (int32_t d = axis + 1; d < rank; ++d)
            oshape[orank++] = shape[d];
    }
    else
    {
        const int32_t width = ishape[irank - 1];
        if (irank - 1 + rank - batch - width > 4)
            return 0;
        for (int32_t d = 0; d < irank - 1; ++d)
            oshape[orank++] = ishape[d];
        for (int32_t d = batch + width; d < rank; ++d)
            oshape[orank++] = shape[d];
    }
    memset(c, 0, sizeof(*c));
    c->kind = kind;
    c->rank = rank;
    c->irank = irank;
    c->axis = negative && !gd_nd(kind) ? axis - rank : axis;
    c->batch = negative && !gd_nd(kind) && irank > 0 ? batch - irank : batch;
    c->in = gd_dims(shape, rank, rest);
    c->idx = gd_dims(ishape, irank, rest);
    c->out = gd_dims(oshape, orank, rest);
    c->out_count = gd_product(oshape, 0, orank);
    const long long count = gd_product(ishape, 0, irank);
    c->index_count = count > GD_IDX ? GD_IDX : (int32_t)count;
    c->pass_input = c->pass_indices = c->pass_params = c->pass_output = 1;
    c->pass_in_dims = c->pass_idx_dims = c->pass_out_dims = 1;
    return 1;
}

/* Valid indices for a built case: for GATHER each in [0, shape[axis]), for GATHER_ND each
 * coordinate d in [0, shape[batch + d]); drawn at random, or the smallest, or the largest. */
static void
gd_indices_fill(const gd_case *c, const int32_t shape[4], int32_t axis, int32_t batch, int mode, uint32_t *state)
{
    for (int32_t i = 0; i < c->index_count; ++i)
    {
        int32_t extent = shape[axis];
        if (gd_nd(c->kind))
        {
            const int32_t width = c->irank > 0 ? (c->irank == 1       ? c->idx.n
                                                      : c->irank == 2 ? c->idx.h
                                                      : c->irank == 3 ? c->idx.w
                                                                      : c->idx.c)
                                               : 1;
            extent = shape[batch + i % width];
        }
        const uint32_t r = gd_next(state);
        gd_indices[GD_GUARD + i] = extent <= 0 ? 0
            : mode == 1                        ? 0
            : mode == 2                        ? extent - 1
                                               : (int32_t)(r % (uint32_t)extent);
    }
}

static void gd_report(const char *what, unsigned long before_cases, unsigned long before_differences)
{
    snprintf(gd_message,
             sizeof(gd_message),
             "differential %s: %lu cases, %lu differences; so far %lu run accepted, %lu refused; nonempty runs "
             "with an inner run under 4: %lu, of 4: %lu, over 4: %lu",
             what,
             gd_cases - before_cases,
             gd_differences - before_differences,
             gd_statuses[1],
             gd_statuses[0],
             gd_inner_runs[0],
             gd_inner_runs[1],
             gd_inner_runs[2]);
    TEST_MESSAGE(gd_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(gd_differences - before_differences));
}

/* Input extents: zero, one, small, a vector's width on each side, and 127, 128 and 129 on each
 * side of the small-extent gate. Index extents stay small so the indices fit their buffer. */
static const int32_t gd_extents[] = {1, 1, 2, 3, 4, 5, 7, 8, 127, 128, 129, 0};
static const int32_t gd_index_extents[] = {1, 2, 3, 4, 0};

/* Every structure of each kind (rank 1 to 4, every axis, every coordinate rank and batch count
 * the entry accepts, in the plain and the negative forms of the fields), each with random
 * shapes that fit the buffers, both input patterns and random, smallest and largest indices. */
static void gd_shapes(void)
{
    const unsigned long before_cases = gd_cases;
    const unsigned long before_differences = gd_differences;
    uint32_t state = UINT32_C(0x2545f491);
    for (int kind = 0; kind < GD_KINDS; ++kind)
        for (int32_t rank = 1; rank <= 4; ++rank)
            for (int32_t irank = gd_nd(kind) ? 1 : 0; irank <= 4; ++irank)
                for (int32_t axis = 0; axis < rank; ++axis)
                    for (int32_t batch = 0; batch <= irank; ++batch)
                    {
                        if (!gd_nd(kind) ? batch > axis : (axis != 0 || batch >= irank || batch >= rank))
                            continue;
                        int built = 0;
                        for (int sample = 0; sample < 800 && built < 48; ++sample)
                        {
                            int32_t shape[4], ishape[4];
                            for (int d = 0; d < 4; ++d)
                            {
                                shape[d] = gd_extents[gd_next(&state) % (sizeof(gd_extents) / sizeof(gd_extents[0]))];
                                ishape[d] = gd_index_extents[gd_next(&state) %
                                                             (sizeof(gd_index_extents) / sizeof(gd_index_extents[0]))];
                            }
                            for (int32_t d = 0; d < batch; ++d)
                                ishape[d] = shape[d] < 5 ? shape[d] : 2;
                            for (int32_t d = 0; d < batch; ++d)
                                shape[d] = ishape[d];
                            if (gd_nd(kind))
                            {
                                const int32_t most = rank - batch;
                                ishape[irank - 1] = 1 + (int32_t)(gd_next(&state) % (uint32_t)most);
                            }
                            gd_case c;
                            if (!gd_build(&c,
                                          kind,
                                          rank,
                                          irank,
                                          axis,
                                          batch,
                                          shape,
                                          ishape,
                                          sample & 1,
                                          sample % 3 == 2 ? 0x7fff : 1))
                                continue;
                            if (gd_product(shape, 0, rank) > GD_IN || gd_product(ishape, 0, irank) > GD_IDX ||
                                c.out_count > GD_OUT)
                                continue;
                            ++built;
                            for (int pattern = 0; pattern < 2; ++pattern)
                            {
                                gd_fill(pattern, gd_width(kind), &state);
                                gd_indices_fill(&c, shape, axis, batch, (sample + pattern) % 3, &state);
                                gd_run(pattern ? "specials" : "random bits", &c);
                            }
                        }
                    }
    gd_report("shapes", before_cases, before_differences);
}

/* Every refusal of the previous bodies, and their empty successes, on a plain and a batched
 * base case of each kind: each pointer missing; ranks, axes, batch counts and GATHER_ND widths
 * out of range; a negative extent at each position of each shape; each output extent off by
 * one; a batch extent that differs; an index out of range (negative, the extent, past it,
 * INT32_MIN and INT32_MAX) first, in the middle and last, for each coordinate; empty inputs,
 * indices and outputs; and counts on each side of INT32_MAX in bytes. */
static void gd_base(gd_case *c, int kind, int batched)
{
    const int32_t shape[4] = {2, 3, 4, 1};
    int32_t ishape[4] = {2, 2, 1, 1};
    if (!gd_nd(kind))
    {
        if (batched)
            ishape[1] = 3;
        gd_build(c, kind, 3, 2, batched ? 2 : 1, batched ? 1 : 0, shape, ishape, 0, 1);
    }
    else
    {
        if (batched)
        {
            const int32_t nd_ishape[4] = {2, 2, 1, 1};
            gd_build(c, kind, 3, 3, 0, 1, shape, nd_ishape, 0, 1);
        }
        else
        {
            const int32_t nd_ishape[4] = {3, 2, 1, 1};
            gd_build(c, kind, 3, 2, 0, 0, shape, nd_ishape, 0, 1);
        }
    }
    uint32_t state = UINT32_C(0x1234567);
    gd_indices_fill(c, shape, gd_nd(kind) ? 0 : c->axis, c->batch, 0, &state);
}

static int32_t *gd_extent(cmsis_nn_dims *d, int position)
{
    return position == 0 ? &d->n : position == 1 ? &d->h : position == 2 ? &d->w : &d->c;
}

static int32_t gd_out_rank(const gd_case *c)
{
    if (!gd_nd(c->kind))
        return c->rank + c->irank - c->batch - 1;
    return c->irank - 1 + c->rank - c->batch - *gd_extent((cmsis_nn_dims *)&c->idx, c->irank - 1);
}

/* A one-dimensional input of `extent` elements (the gathered axis) and `count` indices of value
 * `value`, GATHER or GATHER_ND, or with `extent` as the second of two extents where `rows` is
 * not 1. */
static void gd_large(int kind, int32_t rows, int32_t extent, int32_t count, int32_t value, const char *what)
{
    gd_case c;
    const int32_t shape[4] = {rows, extent, 1, 1};
    const int32_t rank = rows == 1 ? 1 : 2;
    const int32_t one[4] = {extent, 1, 1, 1};
    int32_t ishape[4] = {count, 1, 1, 1};
    if (gd_nd(kind))
    {
        ishape[1] = rank;
        if (!gd_build(&c, kind, rank, 2, 0, 0, rank == 1 ? one : shape, ishape, 0, 1))
            return;
    }
    else if (!gd_build(&c, kind, rank, 1, rank - 1, 0, rank == 1 ? one : shape, ishape, 0, 1))
        return;
    for (int32_t i = 0; i < c.index_count; ++i)
        gd_indices[GD_GUARD + i] = gd_nd(kind) && rank == 2 && (i & 1) == 0 ? 0 : value;
    gd_run(what, &c);
}

static void gd_refusals(void)
{
    const unsigned long before_cases = gd_cases;
    const unsigned long before_differences = gd_differences;
    uint32_t state = UINT32_C(0x9e3779b9);
    gd_fill(0, sizeof(float32_t), &state);
    for (int kind = 0; kind < GD_KINDS; ++kind)
        for (int batched = 0; batched < 2; ++batched)
        {
            gd_case c;
            gd_base(&c, kind, batched);
            gd_run("base", &c);
            for (int which = 0; which < 7; ++which)
            {
                gd_base(&c, kind, batched);
                int *flags[7] = {&c.pass_input,
                                 &c.pass_indices,
                                 &c.pass_params,
                                 &c.pass_output,
                                 &c.pass_in_dims,
                                 &c.pass_idx_dims,
                                 &c.pass_out_dims};
                *flags[which] = 0;
                gd_run("missing pointer", &c);
            }
            const int32_t ranks[] = {0, 5, -1, INT32_MIN, INT32_MAX};
            for (unsigned i = 0; i < sizeof(ranks) / sizeof(ranks[0]); ++i)
            {
                gd_base(&c, kind, batched);
                c.rank = ranks[i];
                gd_run("rank", &c);
                gd_base(&c, kind, batched);
                c.irank = ranks[i];
                gd_run("index rank", &c);
            }
            const int32_t fields[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, INT32_MIN, INT32_MAX};
            for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
            {
                gd_base(&c, kind, batched);
                c.axis = fields[i];
                gd_run("axis", &c);
                gd_base(&c, kind, batched);
                c.batch = fields[i];
                gd_run("batch", &c);
            }
            for (int dims = 0; dims < 3; ++dims)
                for (int position = 0; position < 4; ++position)
                {
                    const int32_t values[] = {-1, INT32_MIN, 0, INT32_MAX};
                    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i)
                    {
                        gd_base(&c, kind, batched);
                        cmsis_nn_dims *d = dims == 0 ? &c.in : dims == 1 ? &c.idx : &c.out;
                        *gd_extent(d, position) = values[i];
                        if (dims == 1 && values[i] >= 0)
                            continue;
                        gd_run("extent", &c);
                    }
                }
            const int32_t orank = gd_out_rank(&c);
            for (int32_t position = 0; position < orank && position < 4; ++position)
                for (int32_t delta = -1; delta <= 1; delta += 2)
                {
                    gd_base(&c, kind, batched);
                    *gd_extent(&c.out, position) += delta;
                    gd_run("output extent", &c);
                }
            if (batched)
            {
                gd_base(&c, kind, batched);
                c.in.n += 1;
                gd_run("batch extent", &c);
            }
            gd_base(&c, kind, batched);
            const int32_t count = c.index_count;
            for (int place = 0; place < 3; ++place)
                for (int bad = 0; bad < 5; ++bad)
                {
                    gd_base(&c, kind, batched);
                    const int32_t at = place == 0 ? 0 : place == 1 ? count / 2 : count - 1;
                    int32_t extent;
                    if (gd_nd(kind))
                    {
                        const int32_t width = *gd_extent(&c.idx, c.irank - 1);
                        extent = *gd_extent(&c.in, c.batch + at % width);
                    }
                    else
                        extent = *gd_extent(&c.in, c.axis);
                    const int32_t values[5] = {-1, extent, extent + 1, INT32_MIN, INT32_MAX};
                    gd_indices[GD_GUARD + at] = values[bad];
                    gd_run("index", &c);
                }
            if (gd_nd(kind))
            {
                const int32_t widths[] = {0, 4, -1, INT32_MAX};
                for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i)
                {
                    gd_base(&c, kind, batched);
                    *gd_extent(&c.idx, c.irank - 1) = widths[i];
                    gd_run("width", &c);
                }
            }
            /* Empty: no indices (the output follows), a zero input extent with and without indices. */
            gd_base(&c, kind, batched);
            c.idx.n = 0;
            c.index_count = 0;
            if (!gd_nd(kind) && !batched)
                c.out.h = 0;
            if (gd_nd(kind))
                c.out.n = 0;
            gd_run("no indices", &c);
            gd_base(&c, kind, batched);
            *gd_extent(&c.in, c.rank - 1) = 0;
            gd_run("zero input extent", &c);
            gd_base(&c, kind, batched);
            *gd_extent(&c.in, c.rank - 1) = 0;
            *gd_extent(&c.out, gd_out_rank(&c) - 1) = 0;
            gd_run("zero input and output extent", &c);
        }
    /* Counts on each side of INT32_MAX in bytes: a gathered axis of 2^29 - 1, 2^29, 2^30 - 1 and
     * 2^30 elements (the binary32 and binary16 limits) and INT32_MAX, read at its first elements
     * so the entries run where the plan accepts; two rows of 46341 (2^31 elements); and outputs of
     * 2^28 and 2^29 elements, which only the prepare steps compare. */
    const int32_t axes[] = {
        INT32_C(536870911), INT32_C(536870912), INT32_C(1073741823), INT32_C(1073741824), INT32_MAX};
    for (int kind = 0; kind < GD_KINDS; ++kind)
    {
        for (unsigned i = 0; i < sizeof(axes) / sizeof(axes[0]); ++i)
        {
            gd_large(kind, 1, axes[i], 1, 0, "large axis");
            gd_large(kind, 1, axes[i], 3, 2, "large axis");
            gd_large(kind, 1, axes[i], 1, axes[i] - 1, "large axis, last element");
            gd_large(kind, 1, axes[i], 1, axes[i], "large axis, index past it");
        }
        gd_large(kind, 46341, 46341, 1, 0, "rows of 46341");
        gd_large(kind, 2, INT32_C(268435456), 1, 0, "two rows of 2^28");
        gd_large(kind, 1, 127, 1, 126, "gate");
        gd_large(kind, 1, 128, 1, 127, "gate");
    }
    for (int kind = 0; kind < GD_KINDS; ++kind)
    {
        if (gd_nd(kind))
            continue;
        /* Outputs of 2^28 and 2^29 elements: an inner run of 2^27 and 2^28 gathered twice. */
        gd_case c;
        const int32_t shape[4] = {2, INT32_C(134217728), 1, 1};
        const int32_t ishape[4] = {2, 1, 1, 1};
        gd_build(&c, kind, 2, 1, 0, 0, shape, ishape, 0, 1);
        gd_indices[GD_GUARD] = 0;
        gd_indices[GD_GUARD + 1] = 1;
        gd_run("output of 2^28", &c);
        const int32_t wider[4] = {2, INT32_C(268435456), 1, 1};
        gd_build(&c, kind, 2, 1, 0, 0, wider, ishape, 0, 1);
        gd_indices[GD_GUARD] = 0;
        gd_indices[GD_GUARD + 1] = 1;
        gd_run("output of 2^29", &c);
    }
    gd_report("refusals", before_cases, before_differences);
    snprintf(gd_message,
             sizeof(gd_message),
             "differential refusals: %lu accepted shapes beyond the buffers left to the prepare comparison",
             gd_planned_only);
    TEST_MESSAGE(gd_message);
}

/* The count step alone: arm_nn_gather_count against gp_gather_count at ranks 0 to 4 and element
 * sizes 1, 2, 4 and 8, over extents on each side of the small-extent gate (127, 128) and of the
 * byte limits (INT32_MAX / 8, / 4, / 2, / 1 and their neighbours), negative extents, and two
 * extents over powers of two and their neighbours with the others 1. A decision differs when
 * the results differ or, where both accept, the counts. */
static unsigned long gd_count_cases;
static unsigned long gd_count_differences;
static unsigned long gd_count_sides[2][2];

static void gd_count_case(const int32_t *shape, int32_t rank, size_t element_size)
{
    size_t fresh = 7, previous = 7;
    const int32_t got = arm_nn_gather_count(shape, rank, element_size, &fresh);
    const int32_t want = gp_gather_count(shape, rank, element_size, &previous);
    int32_t any = 0;
    for (int32_t d = 0; d < rank; ++d)
        any |= shape[d];
    ++gd_count_sides[any >= 0 && any < 128][want != 0];
    ++gd_count_cases;
    if (got != want || (want && fresh != previous))
    {
        if (gd_count_differences < GD_REPORTED)
        {
            snprintf(gd_message,
                     sizeof(gd_message),
                     "count difference: rank %ld size %u {%ld, %ld, %ld, %ld} %ld against %ld",
                     (long)rank,
                     (unsigned)element_size,
                     rank > 0 ? (long)shape[0] : 0L,
                     rank > 1 ? (long)shape[1] : 0L,
                     rank > 2 ? (long)shape[2] : 0L,
                     rank > 3 ? (long)shape[3] : 0L,
                     (long)got,
                     (long)want);
            TEST_MESSAGE(gd_message);
        }
        ++gd_count_differences;
    }
}

static void gd_counts(void)
{
    const int32_t t[] = {-1,
                         INT32_MIN,
                         0,
                         1,
                         2,
                         3,
                         126,
                         127,
                         128,
                         129,
                         255,
                         256,
                         46340,
                         46341,
                         INT32_C(268435455),
                         INT32_C(268435456),
                         INT32_C(536870911),
                         INT32_C(536870912),
                         INT32_C(1073741823),
                         INT32_C(1073741824),
                         INT32_MAX};
    const int n = (int)(sizeof(t) / sizeof(t[0]));
    const size_t sizes[] = {1, 2, 4, 8};
    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
    {
        gd_count_case(t, 0, sizes[s]);
        for (int a = 0; a < n; ++a)
        {
            const int32_t one[1] = {t[a]};
            gd_count_case(one, 1, sizes[s]);
            for (int b = 0; b < n; ++b)
            {
                const int32_t two[2] = {t[a], t[b]};
                gd_count_case(two, 2, sizes[s]);
                for (int c = 0; c < n; ++c)
                {
                    const int32_t three[3] = {t[a], t[b], t[c]};
                    gd_count_case(three, 3, sizes[s]);
                    for (int e = 0; e < n; ++e)
                    {
                        const int32_t four[4] = {t[a], t[b], t[c], t[e]};
                        gd_count_case(four, 4, sizes[s]);
                    }
                }
            }
        }
        int32_t powers[94];
        int count = 0;
        for (int k = 0; k <= 30; ++k)
        {
            const int32_t p = INT32_C(1) << k;
            powers[count++] = p - 1;
            powers[count++] = p;
            powers[count++] = p + 1;
        }
        powers[count++] = INT32_MAX;
        for (int first = 0; first < 4; ++first)
            for (int second = first + 1; second < 4; ++second)
                for (int i = 0; i < count; ++i)
                    for (int j = 0; j < count; ++j)
                    {
                        int32_t four[4] = {1, 1, 1, 1};
                        four[first] = powers[i];
                        four[second] = powers[j];
                        gd_count_case(four, 4, sizes[s]);
                    }
    }
    snprintf(gd_message,
             sizeof(gd_message),
             "differential counts: %lu cases, %lu differences; inside the gate %lu accepted %lu refused, outside %lu "
             "accepted %lu refused",
             gd_count_cases,
             gd_count_differences,
             gd_count_sides[1][1],
             gd_count_sides[1][0],
             gd_count_sides[0][1],
             gd_count_sides[0][0]);
    TEST_MESSAGE(gd_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)gd_count_differences);
}
