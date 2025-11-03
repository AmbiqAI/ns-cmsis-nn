#
# SPDX-FileCopyrightText: Copyright 2019-2022, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SET(ROOT ${CMSIS_PATH})

# Select which parts of the CMSIS-NN must be compiled.
# There are some dependencies between the parts but they are not tracked
# by this cmake. So, enabling some functions may require to enable some
# other ones.
option(CONCATENATION        "Concatenation"         ON)
option(FULLYCONNECTED       "Fully Connected"       ON)
option(CONVOLUTION          "Convolutions"          ON)
option(ACTIVATION           "Activations"           ON)
option(POOLING              "Pooling"               ON)
option(SOFTMAX              "Softmax"               ON)
option(BASICMATHSNN         "Basic Maths for NN"    ON)
option(COMPARISON           "Comparison"            ON)
option(RESHAPE              "Reshape"               ON)
option(SVDF                 "SVDF"                  ON)
option(LSTM                 "LSTM"                  ON)
option(PAD                  "Pad"                   ON)
option(TRANSPOSE            "Transpose"             ON)
option(QUANTIZATION         "Quantization"          ON)
option(STRIDEDSLICE         "Strided Slice"         ON)

option(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY
       "Enable inline‐assembly version of the requantize routine" OFF)

# Always needed if any other module above is on.
option(NNSUPPORT            "NN Support"            ON)

list(APPEND CMAKE_MODULE_PATH Source)

# There is a dependency to CMSIS-Core.
target_include_directories(cmsis-nn PUBLIC "${ROOT}/CMSIS/Core/Include")
target_link_libraries(cmsis-nn PRIVATE m)

if(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
  target_compile_definitions(cmsis-nn
    PUBLIC CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
endif()

if (BASICMATHSNN)
  add_subdirectory(BasicMathFunctions)
endif()

if (COMPARISON)
  add_subdirectory(ComparisonFunctions)
endif()

if (CONCATENATION)
  add_subdirectory(ConcatenationFunctions)
endif()

if (FULLYCONNECTED)
  add_subdirectory(FullyConnectedFunctions)
endif()

if (CONVOLUTION)
  add_subdirectory(ConvolutionFunctions)
endif()

if (ACTIVATION)
  add_subdirectory(ActivationFunctions)
endif()

if (POOLING)
  add_subdirectory(PoolingFunctions)
endif()

if (SOFTMAX)
  add_subdirectory(SoftmaxFunctions)
endif()

if (SVDF)
  add_subdirectory(SVDFunctions)
endif()

if (LSTM)
  add_subdirectory(LSTMFunctions)
endif()

if (RESHAPE)
  add_subdirectory(ReshapeFunctions)
endif()

if (PAD)
  add_subdirectory(PadFunctions)
endif()

if (TRANSPOSE)
  add_subdirectory(TransposeFunctions)
endif()

if (QUANTIZATION)
  add_subdirectory(QuantizationFunctions)
endif()

if (STRIDEDSLICE)
  add_subdirectory(StridedSliceFunctions)
endif()

# Keep NNSUPPORT at the end
if (NNSUPPORT)
  add_subdirectory(NNSupportFunctions)
endif()
