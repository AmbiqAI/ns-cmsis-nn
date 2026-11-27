# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Host planner library (RFC #829): a shared library per target toolchain, built from Planner/sources.txt.
#   cmake -S Planner -B build/plan -DARM_NN_PLAN_TOOLCHAIN=gcc && cmake --build build/plan
# It is not part of the firmware library and never cross-compiles.

cmake_minimum_required(VERSION 3.15)
project(ns_cmsis_nn_plan LANGUAGES C)

if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "the ns-cmsis-nn planner is a host library; configure it without a cross toolchain")
endif()

set(ARM_NN_PLAN_TOOLCHAIN "gcc" CACHE STRING "Target toolchain the planner routes and sizes for: gcc or armclang")
set_property(CACHE ARM_NN_PLAN_TOOLCHAIN PROPERTY STRINGS gcc armclang)
if(ARM_NN_PLAN_TOOLCHAIN STREQUAL "gcc")
    set(_armclang 0)
elseif(ARM_NN_PLAN_TOOLCHAIN STREQUAL "armclang")
    set(_armclang 1)
else()
    message(FATAL_ERROR "ARM_NN_PLAN_TOOLCHAIN must be gcc or armclang, got '${ARM_NN_PLAN_TOOLCHAIN}'")
endif()

get_filename_component(_root "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/sources.txt" _sources REGEX "^[^#]")
list(TRANSFORM _sources PREPEND "${_root}/")

set(_target ns_cmsis_nn_plan_${ARM_NN_PLAN_TOOLCHAIN})
add_library(${_target} SHARED ${_sources})
target_include_directories(${_target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${_root}/Include"
                                      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(${_target} PRIVATE ARM_NN_TOOLCHAIN_ARMCLANG=${_armclang})
set_target_properties(${_target} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${_target} PRIVATE -Wall -Wextra -Werror)
endif()
