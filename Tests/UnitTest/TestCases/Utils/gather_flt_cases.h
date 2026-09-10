/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

#define GF_CAT_(a, b) a##b
#define GF_CAT(a, b) GF_CAT_(a, b)
#define GF_FN(name) GF_CAT(GF_PREFIX, name)
#define GF_CAP 2048
#define GF_GUARD 8

#if GF_F16
typedef float16_t gf_t;
typedef uint16_t gf_bits_t;
    #define GF_ASSERT_BITS TEST_ASSERT_EQUAL_HEX16
#else
typedef float32_t gf_t;
typedef uint32_t gf_bits_t;
    #define GF_ASSERT_BITS TEST_ASSERT_EQUAL_HEX32
#endif

typedef struct
{
    int rank, irank, axis, batch;
    int32_t shape[4], ishape[4];
} gf_case;

static gf_t gf_input[GF_CAP + 2 * GF_GUARD];
static gf_t gf_output[GF_CAP + 2 * GF_GUARD];
static int32_t gf_indices[GF_CAP];

static gf_bits_t gf_bits(const gf_t *p)
{
    gf_bits_t b;
    memcpy(&b, p, sizeof(b));
    return b;
}

static void gf_put(gf_t *p, gf_bits_t b) { memcpy(p, &b, sizeof(b)); }

static int gf_count(const int32_t *shape, int rank)
{
    int n = 1;
    for (int d = 0; d < rank; ++d)
    {
        n *= shape[d];
    }
    return n;
}

static cmsis_nn_dims gf_dims(const int32_t *s, int rank)
{
    cmsis_nn_dims d = {1, 1, 1, 1};
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

static int gf_shape(const gf_case *c, int32_t out[4])
{
    int n = 0;
#if GF_ND
    for (int d = 0; d < c->irank - 1; ++d)
        out[n++] = c->ishape[d];
    for (int d = c->batch + c->ishape[c->irank - 1]; d < c->rank; ++d)
        out[n++] = c->shape[d];
#else
    int axis = c->axis < 0 ? c->axis + c->rank : c->axis;
    int batch = c->batch < 0 ? c->batch + c->irank : c->batch;
    for (int d = 0; d < axis; ++d)
        out[n++] = c->shape[d];
    for (int d = batch; d < c->irank; ++d)
        out[n++] = c->ishape[d];
    for (int d = axis + 1; d < c->rank; ++d)
        out[n++] = c->shape[d];
#endif
    return n;
}

static arm_cmsis_nn_status gf_raw(const gf_case *c,
                                  const gf_t *in,
                                  const int32_t *indices,
                                  gf_t *out,
                                  const cmsis_nn_dims *id,
                                  const cmsis_nn_dims *xd,
                                  const cmsis_nn_dims *od)
{
#if GF_ND
    cmsis_nn_gather_nd_params p = {c->rank, c->irank, c->batch};
#else
    cmsis_nn_gather_params p = {c->axis, c->batch, c->rank, c->irank};
#endif
    return GF_KERNEL(in, id, indices, xd, &p, out, od);
}

static arm_cmsis_nn_status gf_call(const gf_case *c, const gf_t *in, const int32_t *indices, gf_t *out)
{
    int32_t os[4];
    int rank = gf_shape(c, os);
    cmsis_nn_dims id = gf_dims(c->shape, c->rank), xd = gf_dims(c->ishape, c->irank), od = gf_dims(os, rank);
    return gf_raw(c, in, indices, out, &id, &xd, &od);
}

/* Output-coordinate reference, independent of the copy walker. Refs #493. */
static int gf_reference(const gf_case *c, const int32_t *indices, int flat)
{
    int32_t os[4], oc[4] = {0}, ic[4] = {0}, xc[4] = {0};
    int orank = gf_shape(c, os);
    for (int d = orank - 1; d >= 0; --d)
    {
        oc[d] = flat % os[d];
        flat /= os[d];
    }
#if GF_ND
    int width = c->ishape[c->irank - 1];
    int tuple = 0;
    for (int d = 0; d < c->irank - 1; ++d)
        tuple = tuple * c->ishape[d] + oc[d];
    for (int d = 0; d < c->batch; ++d)
        ic[d] = oc[d];
    for (int d = 0; d < width; ++d)
        ic[c->batch + d] = indices[tuple * width + d];
    for (int d = c->batch + width; d < c->rank; ++d)
        ic[d] = oc[c->irank - 1 + d - c->batch - width];
    (void)xc;
#else
    int axis = c->axis < 0 ? c->axis + c->rank : c->axis;
    int batch = c->batch < 0 ? c->batch + c->irank : c->batch;
    for (int d = 0; d < batch; ++d)
        xc[d] = oc[d];
    for (int d = batch; d < c->irank; ++d)
        xc[d] = oc[axis + d - batch];
    int x = 0;
    for (int d = 0; d < c->irank; ++d)
        x = x * c->ishape[d] + xc[d];
    for (int d = 0; d < axis; ++d)
        ic[d] = oc[d];
    ic[axis] = indices[x];
    for (int d = axis + 1; d < c->rank; ++d)
        ic[d] = oc[d + c->irank - batch - 1];
#endif
    int input = 0;
    for (int d = 0; d < c->rank; ++d)
        input = input * c->shape[d] + ic[d];
    return input;
}

static void gf_reset(void)
{
    memset(gf_input, 0xa5, sizeof(gf_input));
    memset(gf_output, 0x5a, sizeof(gf_output));
}

static void gf_unchanged(const gf_t *p, int n, unsigned char value)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t k = 0; k < (size_t)n * sizeof(gf_t); ++k)
        TEST_ASSERT_EQUAL_HEX8(value, b[k]);
}

static void gf_guards(int nin, int nout)
{
    gf_unchanged(gf_input, GF_GUARD, 0xa5);
    gf_unchanged(gf_input + GF_GUARD + nin, GF_CAP + GF_GUARD - nin, 0xa5);
    gf_unchanged(gf_output, GF_GUARD, 0x5a);
    gf_unchanged(gf_output + GF_GUARD + nout, GF_CAP + GF_GUARD - nout, 0x5a);
}

static gf_bits_t gf_pattern(uint32_t k)
{
#if GF_F16
    static const uint16_t special[] = {
        0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x7c00, 0xfc00, 0x7e01, 0xfe55, 0x7c01, 0xfc01, 0x7bff, 0xfbff};
#else
    static const uint32_t special[] = {0,
                                       0x80000000,
                                       1,
                                       0x80000001,
                                       0x007fffff,
                                       0x00800000,
                                       0x7f800000,
                                       0xff800000,
                                       0x7fc00001,
                                       0xffc05555,
                                       0x7f800001,
                                       0xff800001,
                                       0x7f7fffff,
                                       0xff7fffff};
#endif
    if (k % 3 == 0)
        return special[(k / 3) % (sizeof(special) / sizeof(special[0]))];
    return (gf_bits_t)(k * UINT32_C(2654435761) + UINT32_C(0x31415926));
}

static void gf_run(const gf_case *c)
{
    int32_t os[4];
    int orank = gf_shape(c, os);
    int nout = gf_count(os, orank);
    int nin = gf_count(c->shape, c->rank), nx = gf_count(c->ishape, c->irank);
    TEST_ASSERT_LESS_OR_EQUAL(GF_CAP, nin);
    TEST_ASSERT_LESS_OR_EQUAL(GF_CAP, nx);
    TEST_ASSERT_LESS_OR_EQUAL(GF_CAP, nout);
    gf_reset();
    for (int i = 0; i < nin; ++i)
        gf_put(gf_input + GF_GUARD + i, gf_pattern((uint32_t)i));
    for (int i = 0; i < nx; ++i)
    {
#if GF_ND
        int width = c->ishape[c->irank - 1];
        int extent = c->shape[c->batch + i % width];
#else
        int axis = c->axis < 0 ? c->axis + c->rank : c->axis;
        int extent = c->shape[axis];
#endif
        gf_indices[i] = (i * 7 + 1) % extent;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, gf_call(c, gf_input + GF_GUARD, gf_indices, gf_output + GF_GUARD));
    for (int i = 0; i < nout; ++i)
        GF_ASSERT_BITS(gf_pattern((uint32_t)gf_reference(c, gf_indices, i)), gf_bits(gf_output + GF_GUARD + i));
    for (int i = 0; i < nin; ++i)
        GF_ASSERT_BITS(gf_pattern((uint32_t)i), gf_bits(gf_input + GF_GUARD + i));
    gf_guards(nin, nout);
}

void GF_FN(_rank_batch_sweep)(void)
{
    for (int rank = 1; rank <= 4; ++rank)
    {
#if GF_ND
        for (int irank = 1; irank <= 4; ++irank)
            for (int batch = 0; batch < rank && batch < irank; ++batch)
                for (int width = 1; width + batch <= rank; ++width)
                {
                    if (irank - 1 + rank - batch - width > 4)
                        continue;
                    gf_case c = {rank, irank, 0, batch, {2, 3, 2, 3}, {2, 2, 2, 2}};
                    for (int d = 0; d < batch; ++d)
                        c.ishape[d] = c.shape[d];
                    c.ishape[irank - 1] = width;
                    gf_run(&c);
                }
#else
        for (int irank = 0; irank <= 4; ++irank)
            for (int axis = 0; axis < rank; ++axis)
                for (int batch = 0; batch <= irank && batch <= axis; ++batch)
                {
                    if (rank + irank - batch - 1 > 4)
                        continue;
                    gf_case c = {rank, irank, axis, batch, {2, 3, 2, 3}, {2, 2, 2, 2}};
                    for (int d = 0; d < batch; ++d)
                        c.ishape[d] = c.shape[d];
                    gf_run(&c);
                    c.axis -= rank;
                    if (batch < irank)
                        c.batch -= irank;
                    gf_run(&c);
                }
#endif
    }
}

void GF_FN(_tails_and_payloads)(void)
{
    static const int32_t tails[] = {1, 3, 5, 7, 8, 9, 13, 16, 17, 31, 33};
    for (unsigned i = 0; i < sizeof(tails) / sizeof(tails[0]); ++i)
    {
        gf_case c = {3, 2, 1, 1, {2, 5, tails[i], 1}, {2, 3, 1, 1}};
#if GF_ND
        c.irank = 3;
#endif
        gf_run(&c);
    }
#if GF_F16
    /* Seven-element slices exercise every half encoding through copy tails. */
    gf_case c = {2, 1, 0, 0, {37, 7, 1, 1}, {37, 1, 1, 1}};
    #if GF_ND
    c.irank = 2;
    #endif
    for (uint32_t start = 0; start < 65536; start += 259)
    {
        gf_reset();
        for (int i = 0; i < 259; ++i)
            gf_put(gf_input + GF_GUARD + i, (uint16_t)(start + (uint32_t)i));
        for (int i = 0; i < 37; ++i)
            gf_indices[i] = 36 - i;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, gf_call(&c, gf_input + GF_GUARD, gf_indices, gf_output + GF_GUARD));
        for (int i = 0; i < 259; ++i)
            GF_ASSERT_BITS((uint16_t)(start + (uint32_t)gf_reference(&c, gf_indices, i)),
                           gf_bits(gf_output + GF_GUARD + i));
        gf_guards(259, 259);
    }
#endif
}

static void gf_empty(const gf_case *c, const int32_t *indices)
{
    int nin = gf_count(c->shape, c->rank), nx = gf_count(c->ishape, c->irank);
    gf_reset();
    const gf_t *in = nin == 0 ? NULL : gf_input + GF_GUARD;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, gf_call(c, in, indices, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, gf_call(c, in, indices, gf_output + GF_GUARD));
    if (nx == 0)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, gf_call(c, in, NULL, NULL));
    if (nin != 0)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, gf_call(c, NULL, indices, NULL));
    if (nx != 0)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, gf_call(c, in, NULL, NULL));
    gf_unchanged(gf_output, GF_CAP + 2 * GF_GUARD, 0x5a);
    gf_unchanged(gf_input, GF_CAP + 2 * GF_GUARD, 0xa5);
}

void GF_FN(_empty)(void)
{
#if GF_ND
    gf_case c = {2, 2, 0, 0, {2, 3, 1, 1}, {0, 1, 1, 1}};
    gf_empty(&c, NULL);
    c.shape[0] = 0;
    gf_empty(&c, NULL);
    c.shape[0] = 2;
    c.shape[1] = 0;
    gf_empty(&c, NULL);
    c = (gf_case){3, 3, 0, 1, {0, 3, 7, 1}, {0, 2, 1, 1}};
    gf_empty(&c, NULL);
#else
    int32_t indices[] = {1, 0};
    gf_case c = {2, 1, 0, 0, {2, 3, 1, 1}, {0, 1, 1, 1}};
    gf_empty(&c, NULL);
    c = (gf_case){3, 1, 1, 0, {0, 2, 3, 1}, {2, 1, 1, 1}};
    gf_empty(&c, indices);
    c.shape[0] = 3;
    c.shape[2] = 0;
    gf_empty(&c, indices);
    c = (gf_case){2, 2, 1, 1, {0, 3, 1, 1}, {0, 2, 1, 1}};
    gf_empty(&c, NULL);
    c = (gf_case){1, 1, 0, 0, {0, 1, 1, 1}, {0, 1, 1, 1}};
    gf_empty(&c, NULL);
#endif
}

static void gf_bad(const gf_case *c,
                   const gf_t *in,
                   const int32_t *indices,
                   gf_t *out,
                   const cmsis_nn_dims *id,
                   const cmsis_nn_dims *xd,
                   const cmsis_nn_dims *od)
{
    memset(gf_output, 0x5a, sizeof(gf_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, gf_raw(c, in, indices, out, id, xd, od));
    gf_unchanged(gf_output, GF_CAP + 2 * GF_GUARD, 0x5a);
}

void GF_FN(_invalid_indices)(void)
{
    gf_case c = {2, 1, 0, 0, {2, 3, 1, 1}, {3, 1, 1, 1}};
#if GF_ND
    c.irank = 2;
    c.ishape[1] = 2;
    int32_t indices[] = {1, 2, 0, 1, 0, 3};
#else
    int32_t indices[] = {1, 0, 2};
#endif
    int32_t os[4];
    int orank = gf_shape(&c, os);
    cmsis_nn_dims id = gf_dims(c.shape, c.rank), xd = gf_dims(c.ishape, c.irank), od = gf_dims(os, orank);
    int nx = gf_count(c.ishape, c.irank);
    gf_bad(&c, gf_input + GF_GUARD, indices, gf_output + GF_GUARD, &id, &xd, &od);
    indices[nx - 1] = -1;
    gf_bad(&c, gf_input + GF_GUARD, indices, gf_output + GF_GUARD, &id, &xd, &od);
#if GF_ND
    indices[nx - 1] = 0;
    indices[nx - 2] = 2;
    gf_bad(&c, gf_input + GF_GUARD, indices, gf_output + GF_GUARD, &id, &xd, &od);
    c.ishape[1] = 1;
    c.shape[1] = 0;
    indices[0] = indices[1] = indices[2] = 0;
#else
    c.shape[1] = 0;
    indices[nx - 1] = 0;
    indices[0] = 2;
#endif
    orank = gf_shape(&c, os);
    id = gf_dims(c.shape, c.rank);
    xd = gf_dims(c.ishape, c.irank);
    od = gf_dims(os, orank);
    gf_bad(&c, NULL, indices, NULL, &id, &xd, &od);
#if !GF_ND
    c = (gf_case){3, 1, 1, 0, {0, 2, 3, 1}, {3, 1, 1, 1}};
    orank = gf_shape(&c, os);
    id = gf_dims(c.shape, c.rank);
    xd = gf_dims(c.ishape, c.irank);
    od = gf_dims(os, orank);
    gf_bad(&c, NULL, indices, NULL, &id, &xd, &od);
    indices[0] = -1;
    gf_bad(&c, NULL, indices, NULL, &id, &xd, &od);
#endif
}

void GF_FN(_invalid_metadata)(void)
{
    gf_case c = {2, 1, 0, 0, {2, 3, 1, 1}, {2, 1, 1, 1}};
#if GF_ND
    c.irank = 2;
#endif
    int32_t os[4], indices[] = {1, 0};
    int orank = gf_shape(&c, os);
    cmsis_nn_dims id = gf_dims(c.shape, c.rank), xd = gf_dims(c.ishape, c.irank), od = gf_dims(os, orank);
    gf_t *in = gf_input + GF_GUARD, *out = gf_output + GF_GUARD;
    gf_bad(&c, NULL, indices, out, &id, &xd, &od);
    gf_bad(&c, in, NULL, out, &id, &xd, &od);
    gf_bad(&c, in, indices, NULL, &id, &xd, &od);
    gf_bad(&c, in, indices, out, NULL, &xd, &od);
    gf_bad(&c, in, indices, out, &id, NULL, &od);
    gf_bad(&c, in, indices, out, &id, &xd, NULL);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, GF_KERNEL(in, &id, indices, &xd, NULL, out, &od));
    gf_unchanged(gf_output, GF_CAP + 2 * GF_GUARD, 0x5a);
    gf_case bad = c;
    bad.rank = 0;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad.rank = 5;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad = c;
    bad.irank = -1;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad.irank = 5;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
#if GF_ND
    bad.irank = 0;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad = c;
    bad.batch = -1;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad.batch = c.irank;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    cmsis_nn_dims badx = xd;
    badx.h = 0;
    gf_bad(&c, in, indices, out, &id, &badx, &od);
    badx.h = 3;
    gf_bad(&c, in, indices, out, &id, &badx, &od);
    bad = c;
    bad.batch = 1;
    badx.h = 2;
    gf_bad(&bad, in, indices, out, &id, &badx, &od);
#else
    bad = c;
    bad.axis = -3;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad.axis = 2;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad = c;
    bad.batch = -2;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad.batch = 2;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
    bad.batch = 1;
    gf_bad(&bad, in, indices, out, &id, &xd, &od);
#endif
    cmsis_nn_dims badd = id;
    badd.n = -1;
    gf_bad(&c, in, indices, out, &badd, &xd, &od);
    badd = xd;
    badd.n = -1;
    gf_bad(&c, in, indices, out, &id, &badd, &od);
    badd = od;
    ++badd.n;
    gf_bad(&c, in, indices, out, &id, &xd, &badd);
    bad = c;
    bad.batch = 1;
#if !GF_ND
    bad.axis = 1;
#endif
    badd = xd;
    badd.n = 1;
    gf_bad(&bad, in, indices, out, &id, &badd, &od);
    /* Zero extents do not bypass metadata validation. */
    badd = id;
    badd.n = 0;
    badd.h = -1;
    gf_bad(&c, NULL, indices, NULL, &badd, &xd, &od);
}

void GF_FN(_capacity_and_rank)(void)
{
    int32_t indices[] = {0, 0, 0, 0};
    gf_case c = {4, 4, 0, 0, {1, 1, 1, 1}, {1, 1, 1, 1}};
    cmsis_nn_dims ones = {1, 1, 1, 1}, huge = {INT32_MAX, 1, 1, 1};
    gf_t *in = gf_input + GF_GUARD, *out = gf_output + GF_GUARD;
    gf_bad(&c, in, indices, out, &ones, &ones, &ones);
    c.rank = 1;
    c.irank = 1;
#if GF_ND
    c.irank = 2;
#endif
    gf_bad(&c, in, indices, out, &huge, &ones, &ones);
    gf_bad(&c, in, indices, out, &ones, &huge, &huge);
    cmsis_nn_dims large = {32768, 32768, 1, 1};
    c.rank = 2;
    gf_bad(&c, in, indices, out, &large, &ones, &ones);
    c.rank = 4;
    large = (cmsis_nn_dims){65536, 65536, 65536, 65536};
    gf_bad(&c, in, indices, out, &large, &ones, &ones);
    /* Each source buffer fits; only the inferred output exceeds the byte cap. */
    c = (gf_case){2, 1, 0, 0, {1, 65536, 1, 1}, {65536, 1, 1, 1}};
#if GF_ND
    c.irank = 2;
#endif
    cmsis_nn_dims id = {1, 65536, 1, 1}, xd = {65536, 1, 1, 1}, od = {65536, 65536, 1, 1};
    gf_bad(&c, in, indices, out, &id, &xd, &od);
}

#include "gather_flt_oracle_data.h"

void GF_FN(_oracle)(void)
{
    for (unsigned k = 0; k < sizeof(gf_oracle) / sizeof(gf_oracle[0]); ++k)
    {
        const gf_oracle_case *r = gf_oracle + k;
        gf_reset();
        for (int i = 0; i < r->nin; ++i)
            gf_put(gf_input + GF_GUARD + i, r->input[i]);
        cmsis_nn_dims id = gf_dims(r->c.shape, r->c.rank), xd = gf_dims(r->c.ishape, r->c.irank);
        cmsis_nn_dims od = gf_dims(r->oshape, r->orank);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          gf_raw(&r->c, gf_input + GF_GUARD, r->indices, gf_output + GF_GUARD, &id, &xd, &od));
        for (int i = 0; i < r->nout; ++i)
            GF_ASSERT_BITS(r->output[i], gf_bits(gf_output + GF_GUARD + i));
        gf_guards(r->nin, r->nout);
    }
}
