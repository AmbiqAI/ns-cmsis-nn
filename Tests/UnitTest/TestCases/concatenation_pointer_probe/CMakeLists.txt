# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

# GCC accepts the unused out-of-bounds pointer in constant evaluation; use Clang.
find_program(CONCAT_POINTER_CLANGXX NAMES clang++)
if(CONCAT_POINTER_CLANGXX)
    # A separately found Clang may default to another architecture.
    if(CMAKE_LIBRARY_ARCHITECTURE)
        set(_concat_pointer_target "--target=${CMAKE_LIBRARY_ARCHITECTURE}")
    endif()
    add_test(NAME concatenation_pointer_probe
        COMMAND "${CONCAT_POINTER_CLANGXX}" ${_concat_pointer_target} -std=c++14 -fsyntax-only
            -DARM_NN_ENABLE_F32=0 -DARM_NN_ENABLE_F16=0
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../../Include"
            "${CMAKE_CURRENT_SOURCE_DIR}/concatenation_pointer_probe.cpp")
    add_test(NAME concatenation_s8_pointer_probe
        COMMAND "${CONCAT_POINTER_CLANGXX}" ${_concat_pointer_target} -std=c++20 -fsyntax-only
            -DARM_NN_ENABLE_F32=0 -DARM_NN_ENABLE_F16=0
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../../Include"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ConcatenationFunctions"
            "${CMAKE_CURRENT_SOURCE_DIR}/concatenation_s8_pointer_probe.cpp")
else()
    message(STATUS "concatenation pointer probes not registered: clang++ is unavailable")
endif()
