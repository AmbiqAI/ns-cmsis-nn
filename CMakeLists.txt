#
# SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

project(CMSISNN)

add_library(cmsis-nn STATIC)

target_compile_options(cmsis-nn PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

option(CMSIS_NN_HIDE_INTERNAL_SYMBOLS
       "Build cmsis-nn with hidden-by-default visibility for release packaging."
       OFF)
set(CMSIS_NN_PUBLIC_HEADERS_DIR "" CACHE PATH
    "Annotated public header directory used when CMSIS_NN_HIDE_INTERNAL_SYMBOLS is enabled.")

if(CMSIS_NN_HIDE_INTERNAL_SYMBOLS)
  if(CMSIS_NN_PUBLIC_HEADERS_DIR STREQUAL "")
    message(FATAL_ERROR
      "CMSIS_NN_PUBLIC_HEADERS_DIR must be set when CMSIS_NN_HIDE_INTERNAL_SYMBOLS=ON")
  endif()

  target_compile_options(cmsis-nn PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang,ARMClang>:-fvisibility=hidden>)
  target_include_directories(cmsis-nn BEFORE PUBLIC "${CMSIS_NN_PUBLIC_HEADERS_DIR}")
endif()

target_include_directories(cmsis-nn PUBLIC "Include")

add_subdirectory(Source)
