/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_py_batch_matmul_buffer_size.cpp
 * Description:  Batch matmul buffer size pybinds (optional Python module)
 *
 * $Date:        19 Aug 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include <array>
#include <sstream>

#include "arm_py_common.hpp"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace py = pybind11;

void batch_matmul_buffer_size(py::module_ &m)
{
    m.def(
        "batch_matmul_buffer_size",
        [](Backend backend, DataType data_type, const std::array<int32_t, 4> &input_rhs_nhwc) -> int32_t {
            const cmsis_nn_dims input_rhs_dims = make_dims(input_rhs_nhwc);

            switch (data_type)
            {
            case DataType::A8W8:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_batch_matmul_s8_get_buffer_size_mve(&input_rhs_dims);
                case Backend::DSP:
                    return arm_batch_matmul_s8_get_buffer_size_dsp(&input_rhs_dims);
                case Backend::SCALAR:
                    return arm_batch_matmul_s8_get_buffer_size(&input_rhs_dims);
                }
                break;
            }
            std::ostringstream msg;
            msg << "invalid Backend/DataType combination: backend=" << static_cast<int>(backend)
                << " data_type=" << static_cast<int>(data_type);
            throw py::value_error(msg.str());
        },
        py::arg("backend"),
        py::arg("data_type"),
        py::arg("input_rhs_nhwc"));
}
