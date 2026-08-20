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
            assert py == raw
