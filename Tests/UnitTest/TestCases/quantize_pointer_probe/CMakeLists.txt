# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
find_program(QUANTIZE_POINTER_CLANGXX NAMES clang++)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(QUANTIZE_POINTER_CLANGXX)
    if(CMAKE_LIBRARY_ARCHITECTURE)
        set(_quantize_pointer_target "--target=${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    add_test(NAME quantize_tail_pointer_probe
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_probe.py"
            --source "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
            --compiler "${QUANTIZE_POINTER_CLANGXX}" ${_quantize_pointer_target})
else()
    message(STATUS "quantize pointer probe not registered: clang++ is unavailable")
endif()
