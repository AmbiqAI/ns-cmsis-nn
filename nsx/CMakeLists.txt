# ---------------------------------------------------------------------------
# nsx-cmsis-nn — NSX module for Ambiq's ns-cmsis-nn kernels
# ---------------------------------------------------------------------------
# Supports two modes:
#
#   1. Source build (default):
#      Compiles ns-cmsis-nn operator groups via the SSoT module
#      (cmake/ns_cmsis_nn.cmake). Selection is controlled by
#      NSX_CMSIS_NN_GROUPS (default: ALL).
#
#   2. Prebuilt static library:
#      Set NSX_CMSIS_NN_LIB to the path of a prebuilt .a file.
#      Only the public headers are exposed; no sources are compiled.
#
# CMake options:
#   NSX_CMSIS_NN_LIB                         — path to prebuilt .a (optional)
#   NSX_CMSIS_NN_OPTIMIZATION                — compiler optimization (default: -Ofast)
#   NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM   — enable inline-asm requantize (OFF)
#   NSX_CMSIS_NN_GROUPS                      — operator groups to compile in
#                                              source mode (default: ALL; or a
#                                              CMake list of group ids — see
#                                              cmake/ns_cmsis_nn.cmake)
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)
include(GNUInstallDirs)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ns_cmsis_nn.cmake)

if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
    message(FATAL_ERROR
        "Board must define NSX_BOARD_FLAGS_TARGET before nsx-cmsis-nn is added.")
endif()

# --- Configuration options ---
set(NSX_CMSIS_NN_OPTIMIZATION "-Ofast" CACHE STRING
    "Compiler optimization level for ns-cmsis-nn source builds.")
option(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM
    "Enable inline-assembly version of the requantize routine" OFF)
set(NSX_CMSIS_NN_GROUPS "ALL" CACHE STRING
    "Operator groups to compile into nsx_cmsis_nn (ALL or a CMake list of \
group ids; see cmake/ns_cmsis_nn.cmake).")

# --- Prebuilt vs source build ---
if(DEFINED NSX_CMSIS_NN_LIB)
    # ── Prebuilt static library mode ──────────────────────────────────────
    if(NOT EXISTS "${NSX_CMSIS_NN_LIB}")
        message(FATAL_ERROR
            "nsx-cmsis-nn: prebuilt library not found:\n"
            "  ${NSX_CMSIS_NN_LIB}")
    endif()

    message(STATUS "nsx-cmsis-nn: using prebuilt ${NSX_CMSIS_NN_LIB}")

    add_library(nsx_cmsis_nn_prebuilt STATIC IMPORTED GLOBAL)
    set_target_properties(nsx_cmsis_nn_prebuilt PROPERTIES
        IMPORTED_LOCATION "${NSX_CMSIS_NN_LIB}"
    )

    add_library(nsx_cmsis_nn INTERFACE)
    target_link_libraries(nsx_cmsis_nn
        INTERFACE
            ${NSX_BOARD_FLAGS_TARGET}
            nsx_cmsis_nn_prebuilt
    )
    target_include_directories(nsx_cmsis_nn
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../Include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nsx/modules/nsx-cmsis-nn/Include>
    )

else()
    # ── Source build mode ─────────────────────────────────────────────────
    message(STATUS "nsx-cmsis-nn: building from source (groups=${NSX_CMSIS_NN_GROUPS})")

    add_library(nsx_cmsis_nn STATIC)

    ns_cmsis_nn_attach(nsx_cmsis_nn
        GROUPS ${NSX_CMSIS_NN_GROUPS}
        INCLUDE_DIRS_VISIBILITY PUBLIC)

    # SSoT attaches Include/ as a build-tree path; add the install-tree path
    # so downstream consumers of the installed nsx-cmsis-nn keep working.
    target_include_directories(nsx_cmsis_nn PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nsx/modules/nsx-cmsis-nn/Include>
    )

    target_compile_options(nsx_cmsis_nn PRIVATE ${NSX_CMSIS_NN_OPTIMIZATION})

    target_link_libraries(nsx_cmsis_nn
        PUBLIC ${NSX_BOARD_FLAGS_TARGET}
    )
    if(TARGET nsx_soc_hal)
        target_link_libraries(nsx_cmsis_nn PRIVATE nsx_soc_hal)
    endif()

    if(NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM)
        target_compile_definitions(nsx_cmsis_nn
            PUBLIC CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
    endif()

endif()

# --- Common: alias and export ---
set_target_properties(nsx_cmsis_nn PROPERTIES EXPORT_NAME cmsis_nn)
add_library(nsx::cmsis_nn ALIAS nsx_cmsis_nn)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../Include
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nsx/modules/nsx-cmsis-nn)
install(TARGETS nsx_cmsis_nn EXPORT nsxTargets)
