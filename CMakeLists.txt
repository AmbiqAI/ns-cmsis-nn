#
# SPDX-FileCopyrightText: Copyright 2010-2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
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

cmake_minimum_required(VERSION 3.15.6)
set(CMSIS_OPTIMIZATION_LEVEL "-Ofast" CACHE STRING "Compiler optimization level.")
option(ARM_NN_ENABLE_F32 "Enable CMSIS-NN float32 extensions." OFF)
option(ARM_NN_ENABLE_F16 "Enable CMSIS-NN float16 extensions." OFF)

project(CMSISNN C CXX)

# Single source of truth for the operator-group source layout.
# Consumed by this standalone build, by zephyr/CMakeLists.txt, and by
# nsx/CMakeLists.txt.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ns_cmsis_nn.cmake)

# ----------------------------------------------------------------------------
# Operator-group enable knobs.
#
# These mirror the legacy option() names so that existing invocations like
# `cmake -DCONVOLUTION=OFF ...` continue to work. Internally they are
# translated to a GROUPS list passed into ns_cmsis_nn_attach().
# ----------------------------------------------------------------------------
option(ACTIVATION      "Activations"        ON)
option(BASICMATHSNN    "Basic Maths for NN" ON)
option(COMPARISON      "Comparison"         ON)
option(CONCATENATION   "Concatenation"      ON)
option(CONVOLUTION     "Convolutions"       ON)
option(FULLYCONNECTED  "Fully Connected"    ON)
option(GATHER          "Gather"             ON)
option(LSTM            "LSTM"               ON)
option(PAD             "Pad"                ON)
option(POOLING         "Pooling"            ON)
option(QUANTIZATION    "Quantization"       ON)
option(RESHAPE         "Reshape"            ON)
option(SOFTMAX         "Softmax"            ON)
option(STRIDEDSLICE    "Strided Slice"      ON)
option(SVDF            "SVDF"               ON)
option(TRANSPOSE       "Transpose"          ON)
# Always needed if any other module above is on.
option(NNSUPPORT       "NN Support"         ON)

option(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY
       "Enable inline-assembly version of the requantize routine" OFF)

# Translate the per-group options into a GROUPS list for the SSoT module.
set(_NS_ENABLED_GROUPS "")
foreach(_pair
    "ACTIVATION;activation"
    "BASICMATHSNN;basicmath"
    "COMPARISON;comparison"
    "CONCATENATION;concatenation"
    "CONVOLUTION;convolution"
    "FULLYCONNECTED;fullyconnected"
    "GATHER;gather"
    "LSTM;lstm"
    "PAD;pad"
    "POOLING;pooling"
    "QUANTIZATION;quantization"
    "RESHAPE;reshape"
    "SOFTMAX;softmax"
    "STRIDEDSLICE;stridedslice"
    "SVDF;svd"
    "TRANSPOSE;transpose"
    "NNSUPPORT;nnsupport"
    )
  list(GET _pair 0 _opt)
  list(GET _pair 1 _grp)
  if(${_opt})
    list(APPEND _NS_ENABLED_GROUPS "${_grp}")
  endif()
endforeach()

# ----------------------------------------------------------------------------
# Library target.
#
# Primary name is `cmsis-nn` (the historical / upstream name) so that
# downstream consumers that mutate the target with target_compile_options(),
# target_compile_definitions(), or other non-read-only target_*() calls keep
# working unchanged. (CMake disallows those calls on ALIAS targets, so the
# real target must carry the name they reference.)
#
# `ns-cmsis-nn` and `ns::cmsis-nn` are ALIAS targets pointing at it. They are
# usable in read-only contexts (target_link_libraries(), add_dependencies(),
# right-hand side of set_property(), ...).
# ----------------------------------------------------------------------------
add_library(cmsis-nn STATIC)
add_library(ns-cmsis-nn  ALIAS cmsis-nn)
add_library(ns::cmsis-nn ALIAS cmsis-nn)

if(ARM_NN_ENABLE_F32)
    set(ARM_NN_ENABLE_F32_VALUE 1)
else()
    set(ARM_NN_ENABLE_F32_VALUE 0)
endif()

if(ARM_NN_ENABLE_F16)
    set(ARM_NN_ENABLE_F16_VALUE 1)
else()
    set(ARM_NN_ENABLE_F16_VALUE 0)
endif()

target_compile_options(cmsis-nn PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})
target_compile_definitions(cmsis-nn
                           PUBLIC ARM_NN_ENABLE_F32=${ARM_NN_ENABLE_F32_VALUE}
                                  ARM_NN_ENABLE_F16=${ARM_NN_ENABLE_F16_VALUE})

ns_cmsis_nn_attach(cmsis-nn
                   GROUPS ${_NS_ENABLED_GROUPS}
                   INCLUDE_DIRS_VISIBILITY PUBLIC)

# Standalone-build-only dependencies (Zephyr / NSX provide their own).
if(DEFINED CMSIS_PATH)
  target_include_directories(cmsis-nn PUBLIC "${CMSIS_PATH}/CMSIS/Core/Include")
endif()
target_link_libraries(cmsis-nn PRIVATE m)

if(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
  target_compile_definitions(cmsis-nn
    PUBLIC CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
endif()

option(CMSISNN_BUILD_PYBIND "Build pybind11 Python module for CMSIS-NN helpers" OFF)

if(CMSISNN_BUILD_PYBIND)

  # Add shared library for testing bindings with ctest
  add_library(cmsis-nn-shared SHARED
    $<TARGET_PROPERTY:cmsis-nn,SOURCES>
  )
  set_target_properties(cmsis-nn-shared PROPERTIES OUTPUT_NAME cmsis-nn)
  target_include_directories(cmsis-nn-shared PRIVATE
    $<TARGET_PROPERTY:cmsis-nn,INCLUDE_DIRECTORIES>
  )
  target_compile_options(cmsis-nn-shared PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

  set(Python3_FIND_VIRTUALENV FIRST)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_PYTHON_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  find_package(pybind11 CONFIG
              HINTS "${pybind11_PYTHON_CMAKE_DIR}"
  )
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Install pybind11, set pybind11_DIR, or set CMSISNN_BUILD_PYBIND=OFF.")
  endif()

  pybind11_add_module(cmsis_nn MODULE
    Source/Bindings/arm_py_module.cpp
    Source/Bindings/arm_py_backend.cpp
    Source/Bindings/arm_py_avgpool_buffer_size.cpp
    Source/Bindings/arm_py_conv_buffer_size.cpp
    Source/Bindings/arm_py_depthwise_conv_buffer_size.cpp
    Source/Bindings/arm_py_fully_connected_buffer_size.cpp
    Source/Bindings/arm_py_svdf_buffer_size.cpp
    Source/Bindings/arm_py_transpose_conv_buffer_size.cpp
  )
  set_target_properties(cmsis_nn PROPERTIES PREFIX "")
  target_include_directories(cmsis_nn PRIVATE
      ${Python3_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/Include
  )

  target_link_libraries(cmsis_nn PRIVATE
      cmsis-nn
      pybind11::headers
  )

  target_compile_options(cmsis_nn PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

  # Put the built .so into the python package directory inside the wheel
  install(TARGETS cmsis_nn
                  LIBRARY DESTINATION cmsis_nn
                  RUNTIME DESTINATION cmsis_nn
  )

endif()
