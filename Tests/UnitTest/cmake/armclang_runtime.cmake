# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)
set(_cmsis_nn_armclang_errno_source "${CMAKE_CURRENT_LIST_DIR}/armclang_errno.c")

# Test-only binding for Arm kernel/math code inside a single-thread GNU harness.
function(cmsis_nn_link_armclang_test_runtime target root)
    if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "The ACfE test runtime requires a GNU cross-compiled harness.")
    endif()
    get_target_property(_imported ${target} IMPORTED)
    if(NOT _imported)
        message(FATAL_ERROR "The ACfE test runtime requires an imported kernel archive.")
    endif()

    set(_fp)
    set(_members)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m4")
        set(_math "${root}/lib/armlib/m_wm.l")
        set(_fp "${root}/lib/armlib/fz_wm.l")
        set(_members frnd.o fnaninf.o usenofp.o)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m55")
        set(_math "${root}/lib/armlib/m_wv.l")
    else()
        message(FATAL_ERROR "The ACfE test runtime supports only cortex-m4 and cortex-m55.")
    endif()
    set(_c "${root}/lib/armlib/c_wu.l")
    foreach(_archive IN ITEMS "${_math}" "${_c}" ${_fp})
        if(NOT EXISTS "${_archive}")
            message(FATAL_ERROR "Missing ACfE test runtime archive: ${_archive}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_archive}")
    endforeach()

    # Import only the native dependency closure, rather than Arm's complete C
    # runtime or competing AEABI implementations from its floating-point archive.
    set(_dir "${CMAKE_CURRENT_BINARY_DIR}/armclang-test-runtime")
    file(MAKE_DIRECTORY "${_dir}")
    set(_objects)
    foreach(_member IN ITEMS _rserrno.o ${_members})
        if(_member STREQUAL "_rserrno.o")
            set(_archive "${_c}")
        else()
            set(_archive "${_fp}")
        endif()
        set(_object "${_dir}/${_member}")
        execute_process(COMMAND "${CMAKE_AR}" p "${_archive}" "${_member}"
            OUTPUT_FILE "${_object}" ERROR_VARIABLE _error RESULT_VARIABLE _result)
        file(SIZE "${_object}" _size)
        if(NOT _result EQUAL 0 OR _size EQUAL 0)
            message(FATAL_ERROR "Cannot extract ${_archive}(${_member}): ${_error}")
        endif()
        list(APPEND _objects "${_object}")
    endforeach()

    add_library(cmsis_nn_test_armclang_errno STATIC "${_cmsis_nn_armclang_errno_source}")
    target_compile_options(cmsis_nn_test_armclang_errno PRIVATE
        "-mcpu=${CMAKE_SYSTEM_PROCESSOR}" -mthumb -mfloat-abi=hard)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m4")
        target_compile_options(cmsis_nn_test_armclang_errno PRIVATE -mfpu=fpv4-sp-d16)
    endif()
    target_link_libraries(${target} INTERFACE
        "${_math}" ${_objects} cmsis_nn_test_armclang_errno m)
endfunction()
