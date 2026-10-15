# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
find_program(VECTOR_SVDF_POINTER_CLANGXX NAMES clang++)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(VECTOR_SVDF_POINTER_CLANGXX)
    if(CMAKE_LIBRARY_ARCHITECTURE)
        set(_vector_svdf_pointer_target "--target=${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    add_test(NAME vector_svdf_tail_pointer_probe
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_probe.py"
            --source "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
            --compiler "${VECTOR_SVDF_POINTER_CLANGXX}" ${_vector_svdf_pointer_target})
else()
    message(STATUS "vector_svdf pointer probe not registered: clang++ is unavailable")
endif()
