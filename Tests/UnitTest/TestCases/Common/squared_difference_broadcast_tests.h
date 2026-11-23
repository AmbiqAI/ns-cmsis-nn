/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

/* Included by the F16 and F32 suites with the corresponding element type and kernel. */
static int broadcast_offset(cmsis_nn_dims d, int n, int h, int w, int c)
{
    return (((n % d.n) * d.h + h % d.h) * d.w + w % d.w) * d.c + c % d.c;
}

static void squared_difference_broadcast_shapes(void)
{
    const cmsis_nn_dims shape = {2, 2, 3, 9};
    enum { count = 2 * 2 * 3 * 9 };
    broadcast_element a[count], b[count], out[count + 2];
    for (int i = 0; i < count; ++i)
    {
        a[i] = (broadcast_element)(i % 19 - 9);
        b[i] = (broadcast_element)(i % 11 - 5);
    }
    /* Per axis, either both extents match or one of the two inputs has extent 1. */
    for (int pattern = 0; pattern < 81; ++pattern)
    {
        int left[4], right[4], code = pattern;
        const int extents[4] = {shape.n, shape.h, shape.w, shape.c};
        for (int axis = 0; axis < 4; ++axis, code /= 3)
        {
            left[axis] = code % 3 == 1 ? 1 : extents[axis];
            right[axis] = code % 3 == 2 ? 1 : extents[axis];
        }
        const cmsis_nn_dims ad = {left[0], left[1], left[2], left[3]};
        const cmsis_nn_dims bd = {right[0], right[1], right[2], right[3]};
        out[0] = out[count + 1] = (broadcast_element)-1;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, broadcast_kernel(a, &ad, b, &bd, out + 1, &shape));
        int i = 1;
        for (int n = 0; n < shape.n; ++n)
            for (int h = 0; h < shape.h; ++h)
                for (int w = 0; w < shape.w; ++w)
                    for (int c = 0; c < shape.c; ++c, ++i)
                    {
                        int diff = (int)a[broadcast_offset(ad, n, h, w, c)] - (int)b[broadcast_offset(bd, n, h, w, c)];
                        TEST_ASSERT_EQUAL_FLOAT((float)(diff * diff), (float)out[i]);
                    }
        TEST_ASSERT_EQUAL_FLOAT(-1.0f, (float)out[0]);
        TEST_ASSERT_EQUAL_FLOAT(-1.0f, (float)out[count + 1]);
    }
}

static void squared_difference_broadcast_errors(void)
{
    const cmsis_nn_dims one = {1, 1, 1, 1}, two = {1, 1, 1, 2}, three = {1, 1, 1, 3};
    const cmsis_nn_dims zero = {1, 0, 1, 1}, negative = {-1, 1, 1, 1}, huge = {65536, 65536, 1, 1};
    const broadcast_element in = 1;
    broadcast_element out = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(NULL, &one, &in, &one, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, NULL, &in, &one, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &one, NULL, &one, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &one, &in, NULL, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &one, &in, &one, NULL, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &one, &in, &one, &out, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &two, &in, &three, &out, &three));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &one, &in, &two, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &zero, &in, &one, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &one, &in, &negative, &out, &one));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, broadcast_kernel(&in, &huge, &in, &one, &out, &huge));
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, (float)out);
}
