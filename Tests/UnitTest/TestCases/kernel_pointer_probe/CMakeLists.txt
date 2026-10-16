# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

# GCC accepts unused out-of-bounds pointers in constant evaluation; use Clang.
find_program(CONCAT_POINTER_CLANGXX NAMES clang++)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(CONCAT_POINTER_CLANGXX)
    # A separately found Clang may default to another architecture.
    if(CMAKE_LIBRARY_ARCHITECTURE)
        set(_kernel_pointer_target "--target=${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    set(_kernel_pointer_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
    add_test(NAME row_broadcast_s8_pointer_probe
        COMMAND "${CONCAT_POINTER_CLANGXX}" ${_kernel_pointer_target} -std=c++20 -fsyntax-only
            -DARM_NN_ENABLE_F32=0 -DARM_NN_ENABLE_F16=0
            "-I${_kernel_pointer_root}/Include"
            "-I${_kernel_pointer_root}/Source/BasicMathFunctions"
            "${CMAKE_CURRENT_SOURCE_DIR}/row_broadcast_s8_pointer_probe.cpp")
    foreach(_case scalar dsp mve f16_scalar f32_scalar depthwise depthwise_gnu)
        add_test(NAME kernel_pointer_${_case}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_probe.py"
                --source "${_kernel_pointer_root}"
                --compiler "${CONCAT_POINTER_CLANGXX}" ${_kernel_pointer_target} --case "${_case}")
    endforeach()
else()
    message(STATUS "kernel pointer probes not registered: clang++ is unavailable")
endif()
