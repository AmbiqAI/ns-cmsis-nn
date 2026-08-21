# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.

import ctypes
import logging

import pytest

from .test_bindings_common import (
    SHARED_LIB,
    CmsisNnDims,
    get_buffer_size_wrapper_name,
    make_dims,
)

from cmsis_nn import batch_matmul_buffer_size, Backend, DataType

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "input_rhs_nhwc",
    [
        (1, 1, 16, 4),
        # w < c, so a sizer that read ->c instead of ->w would over-report
        # here and under-report above -- neither shape alone pins the field.
        (1, 1, 4, 16),
        # The kernel-facing shape the C unit tests actually use.
        (1, 1, 24, 16),
        # Boundary: no RHS rows means no kernel-sum buffer.
        (1, 1, 0, 8),
    ],
)
def test_batch_matmul_buffer_size_matches_raw(input_rhs_nhwc):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")

    lib = ctypes.CDLL(str(SHARED_LIB))
    argtypes = [ctypes.POINTER(CmsisNnDims)]
    input_rhs_dims = make_dims(input_rhs_nhwc)

    for backend in Backend.__members__.values():
        for data_type in DataType.__members__.values():
            if data_type != DataType.A8W8:
                continue
            func_name = get_buffer_size_wrapper_name("batch_matmul", backend, data_type)
            if not func_name:
                raise RuntimeError(f"No raw function mapping for {backend} {data_type}")
            try:
                raw_func = getattr(lib, func_name)
            except AttributeError:
                raise RuntimeError(f"Missing symbol {func_name} in {SHARED_LIB}")

            raw_func.argtypes = argtypes
            raw_func.restype = ctypes.c_int32

            raw = raw_func(ctypes.byref(input_rhs_dims))
            py = batch_matmul_buffer_size(
                backend,
                data_type,
                input_rhs_nhwc=input_rhs_nhwc,
            )
            logger.debug(
                "Comparing raw C func %s with python buffer size=%d raw buffer size=%d (%s %s)",
                func_name,
                py,
                raw,
                backend,
                data_type,
            )
            # `py == raw` alone is self-consistency: it compares the binding
            # against the very function it wraps, so a sizer computing the
            # wrong thing -- reading input_rhs_dims->c instead of ->w, say --
            # satisfies it with both sides equally wrong. Pin the value the
            # contract requires, derived here rather than obtained from the
            # code under test. The C unit suite pins the same formula in
            # Tests/UnitTest/TestCases/test_arm_batch_matmul_s8; this brings
            # the binding test to that bar.
            rhs_w = input_rhs_nhwc[2]
            mve_expected = rhs_w * ctypes.sizeof(ctypes.c_int32)

            if backend == Backend.MVE:
                # One int32 kernel-sum accumulator per RHS row; the RHS row
                # count is `w`, not `c`.
                assert raw == mve_expected, (
                    f"{func_name}{input_rhs_nhwc} returned {raw}, expected "
                    f"{mve_expected} (w={rhs_w} rows x 4 bytes)"
                )
            elif backend == Backend.DSP:
                # No kernel-sum buffer is used off the MVE path.
                assert raw == 0, f"{func_name}{input_rhs_nhwc} returned {raw}, expected 0"
            else:
                # The plain dispatcher resolves at compile time, so it must
                # equal whichever of the two above matches how this library
                # was built (ARM_MATH_MVEI or not).
                assert raw in (mve_expected, 0), (
                    f"{func_name}{input_rhs_nhwc} returned {raw}, expected "
                    f"{mve_expected} (MVE build) or 0 (non-MVE build)"
                )

            assert py == raw
