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
#   NSX_CMSIS_NN_MANIFEST                    — explicit path to the SDK
#                                              manifest.json describing
#                                              NSX_CMSIS_NN_LIB's contents
#                                              (optional; see discovery
#                                              rules below).
#   NSX_CMSIS_NN_OPTIMIZATION                — compiler optimization (default: -Ofast)
#   NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM   — enable inline-asm requantize (OFF)
#   NSX_CMSIS_NN_ENABLE_F32                  — enable FP32 kernel declarations (OFF).
#                                              In source mode this also compiles the
#                                              kernels; in prebuilt mode it is
#                                              validated against the archive's
#                                              manifest when one can be found.
#   NSX_CMSIS_NN_ENABLE_F16                  — enable FP16 kernel declarations (OFF).
#                                              Same source/prebuilt caveat as above.
#   NSX_CMSIS_NN_GROUPS                      — operator groups to compile in
#                                              source mode (default: ALL; or a
#                                              CMake list of group ids — see
#                                              cmake/ns_cmsis_nn.cmake)
#
# Prebuilt manifest discovery (validates NSX_CMSIS_NN_ENABLE_F32/F16
# against what the archive actually contains):
#   1. NSX_CMSIS_NN_MANIFEST, if explicitly set.
#   2. "<dir of NSX_CMSIS_NN_LIB>/manifest.json" (archive given as a bare
#      lib/ path from an extracted tarball).
#   3. "<dir of NSX_CMSIS_NN_LIB>/../manifest.json" (archive given as the
#      tarball's own lib/libns-cmsis-nn.a, sibling to root manifest.json).
#   If none of these exist, validation is skipped (backward compatible
#   with hand-supplied .a files that have no manifest); requested flags
#   are trusted as-is and a status message notes they are unverified.
# ---------------------------------------------------------------------------

# string(JSON ...) (used for prebuilt manifest validation) requires CMake
# >= 3.19; bumped from 3.15 for this reason.
cmake_minimum_required(VERSION 3.19)
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
option(NSX_CMSIS_NN_ENABLE_F32
    "Enable experimental FP32 kernels in source builds" OFF)
option(NSX_CMSIS_NN_ENABLE_F16
    "Enable experimental FP16 kernels in source builds" OFF)
set(NSX_CMSIS_NN_MANIFEST "" CACHE FILEPATH
    "Explicit path to the manifest.json describing NSX_CMSIS_NN_LIB's \
contents (optional; auto-discovered next to the library otherwise).")
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

    # ARM_NN_ENABLE_F32/F16 gate the float *declarations* in the public
    # headers (Include/Internal/arm_nn_config.h), not just internal
    # implementation choices baked into the archive, so they must be
    # forwarded to consumers here too, mirroring the source-mode bridge below.
    #
    # When a manifest.json for NSX_CMSIS_NN_LIB can be found (or is given
    # explicitly via NSX_CMSIS_NN_MANIFEST), requested flags are validated
    # against the archive's recorded capabilities and a FATAL_ERROR is
    # raised for any request the archive doesn't actually implement —
    # this turns a predictable undefined-reference link error into a
    # clear configure-time diagnostic. If no manifest can be found (e.g. a
    # hand-supplied .a with no accompanying SDK package), validation is
    # skipped and the requested flags are trusted as-is, same as before.
    set(_nsx_manifest "")
    if(NSX_CMSIS_NN_MANIFEST)
        set(_nsx_manifest "${NSX_CMSIS_NN_MANIFEST}")
    else()
        get_filename_component(_nsx_lib_dir "${NSX_CMSIS_NN_LIB}" DIRECTORY)
        if(EXISTS "${_nsx_lib_dir}/manifest.json")
            set(_nsx_manifest "${_nsx_lib_dir}/manifest.json")
        elseif(EXISTS "${_nsx_lib_dir}/../manifest.json")
            get_filename_component(_nsx_manifest "${_nsx_lib_dir}/../manifest.json" ABSOLUTE)
        endif()
        unset(_nsx_lib_dir)
    endif()

    set(_nsx_has_f32 TRUE)
    set(_nsx_has_f16 TRUE)
    if(_nsx_manifest)
        file(READ "${_nsx_manifest}" _nsx_manifest_json)
        string(JSON _nsx_schema_version ERROR_VARIABLE _nsx_json_err
               GET "${_nsx_manifest_json}" schema_version)
        string(JSON _nsx_features ERROR_VARIABLE _nsx_json_err
               GET "${_nsx_manifest_json}" features)
        if(_nsx_features STREQUAL "NOTFOUND" OR NOT _nsx_schema_version OR _nsx_schema_version LESS 2)
            # Conservative fallback for older (pre-features) manifests:
            # treat the archive as integer-only.
            message(STATUS
                "nsx-cmsis-nn: ${_nsx_manifest} has no schema_version>=2 "
                "'features' block; treating archive as integer-only.")
            set(_nsx_has_f32 FALSE)
            set(_nsx_has_f16 FALSE)
        else()
            string(JSON _nsx_has_f32 GET "${_nsx_manifest_json}" features f32)
            string(JSON _nsx_has_f16 GET "${_nsx_manifest_json}" features f16)
            message(STATUS
                "nsx-cmsis-nn: validated ${NSX_CMSIS_NN_LIB} against "
                "${_nsx_manifest} (f32=${_nsx_has_f32} f16=${_nsx_has_f16})")
        endif()

        if(NSX_CMSIS_NN_ENABLE_F32 AND NOT _nsx_has_f32)
            message(FATAL_ERROR
                "nsx-cmsis-nn: NSX_CMSIS_NN_ENABLE_F32 was requested but "
                "the prebuilt archive's manifest (${_nsx_manifest}) does "
                "not report f32 support. Use an archive built with "
                "--enable-f32, or disable NSX_CMSIS_NN_ENABLE_F32.")
        endif()
        if(NSX_CMSIS_NN_ENABLE_F16 AND NOT _nsx_has_f16)
            message(FATAL_ERROR
                "nsx-cmsis-nn: NSX_CMSIS_NN_ENABLE_F16 was requested but "
                "the prebuilt archive's manifest (${_nsx_manifest}) does "
                "not report f16 support. Use an archive built with "
                "--enable-f16, or disable NSX_CMSIS_NN_ENABLE_F16.")
        endif()
    else()
        message(STATUS
            "nsx-cmsis-nn: no manifest.json found for ${NSX_CMSIS_NN_LIB}; "
            "NSX_CMSIS_NN_ENABLE_F32/F16 requests are UNVERIFIED against "
            "archive contents (set NSX_CMSIS_NN_MANIFEST to enable validation).")
    endif()
    unset(_nsx_manifest)

    set(_nsx_f32 0)
    set(_nsx_f16 0)
    if(NSX_CMSIS_NN_ENABLE_F32)
        set(_nsx_f32 1)
    endif()
    if(NSX_CMSIS_NN_ENABLE_F16)
        set(_nsx_f16 1)
    endif()
    target_compile_definitions(nsx_cmsis_nn
        INTERFACE
            ARM_NN_ENABLE_F32=${_nsx_f32}
            ARM_NN_ENABLE_F16=${_nsx_f16}
    )

else()
    # ── Source build mode ─────────────────────────────────────────────────
    message(STATUS "nsx-cmsis-nn: building from source (groups=${NSX_CMSIS_NN_GROUPS})")

    # Bridge NSX float options to the SSoT float gates.
    set(ARM_NN_ENABLE_F32 0)
    set(ARM_NN_ENABLE_F16 0)
    if(NSX_CMSIS_NN_ENABLE_F32)
        set(ARM_NN_ENABLE_F32 1)
    endif()
    if(NSX_CMSIS_NN_ENABLE_F16)
        set(ARM_NN_ENABLE_F16 1)
    endif()

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

    target_compile_definitions(nsx_cmsis_nn
        PUBLIC
            ARM_NN_ENABLE_F32=${ARM_NN_ENABLE_F32}
            ARM_NN_ENABLE_F16=${ARM_NN_ENABLE_F16}
    )

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
