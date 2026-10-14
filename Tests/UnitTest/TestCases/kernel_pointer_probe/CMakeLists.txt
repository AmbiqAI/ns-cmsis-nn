# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

find_program(KERNEL_POINTER_CLANGXX NAMES clang++)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(KERNEL_POINTER_CLANGXX)
    if(CMAKE_LIBRARY_ARCHITECTURE)
        set(_kernel_pointer_target "--target=${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    foreach(_case scalar dsp mve f16 f32)
        add_test(NAME kernel_pointer_${_case}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_probe.py"
                --source "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
                --compiler "${KERNEL_POINTER_CLANGXX}" ${_kernel_pointer_target} --case "${_case}")
    endforeach()
else()
    message(STATUS "kernel pointer probes not registered: clang++ is unavailable")
endif()
