cmake_minimum_required(VERSION 3.15.6)

# Build bare-metal tests for Arm FVP Corstone-300 with optional gcov coverage
project(helia_aot_fvp_tests C)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMSIS_PATH "artifacts/downloads/CMSIS_5" CACHE PATH "Path to CMSIS (CMSIS_5 checkout)")
set(CMSIS_NN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "ns-cmsis-nn repo root")
set(FVP_CORSTONE_300_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300" CACHE PATH "FVP startup/retarget files")
set(GENERATED_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/generated_tests" CACHE PATH "Path to generated tests")

function(_strip_all_optimizations)
  foreach(lang C CXX ASM)
    foreach(var
      CMAKE_${lang}_FLAGS
      CMAKE_${lang}_FLAGS_RELEASE
      CMAKE_${lang}_FLAGS_RELWITHDEBINFO
      CMAKE_${lang}_FLAGS_MINSIZEREL
      CMAKE_${lang}_FLAGS_DEBUG
    )
      if(DEFINED ${var})
        set(_tmp "${${var}}")
        string(REGEX REPLACE "([ ;]|^)-Ofast([ ;]|$)" " " _tmp "${_tmp}")
        string(REGEX REPLACE "([ ;]|^)-O[0-3s]([ ;]|$)" " " _tmp "${_tmp}")
        string(REGEX REPLACE "[ ]+" " " _tmp "${_tmp}")
        string(STRIP "${_tmp}" _tmp)
        set(${var} "${_tmp}" CACHE STRING "" FORCE)
      endif()
    endforeach()
  endforeach()
endfunction()


# Write .gcda to the build tree so each config keeps its own artifacts
# set(COVERAGE_DIR "${CMAKE_BINARY_DIR}/coverage" CACHE PATH "Where .gcda is written (-fprofile-dir)")

if(NOT EXISTS "${CMSIS_PATH}/CMSIS/Core/Include/cmsis_compiler.h")
  message(FATAL_ERROR "CMSIS_PATH does not look valid: ${CMSIS_PATH}")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR)
  message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR not set (e.g. cortex-m55|cortex-m33|cortex-m4)")
endif()

# ---------- Global warnings ----------
add_compile_options(-Wimplicit-function-declaration -Wunused-variable -Wunused-function -Wno-redundant-decls -Wvla)

# ---------- CPU / FPU flags (compile + link) ----------
# Apply once globally; all targets are Cortex-M
add_compile_options(-mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb)
add_link_options(   -mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m4")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv4-sp-d16)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m7")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv5-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv5-d16)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m33" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m55")
  add_compile_options(-mfloat-abi=hard -mfpu=fpv5-sp-d16)
  add_link_options(   -mfloat-abi=hard -mfpu=fpv5-sp-d16)
else()
  add_compile_options(-mfloat-abi=soft)
  add_link_options(   -mfloat-abi=soft)
endif()

# ---------- cmsis-nn subproject ----------
if(EXISTS "${CMSIS_NN_ROOT}/CMakeLists.txt")
  # Use a stable binary subdir name "cmsis-nn" (we need this exact name below)
  add_subdirectory("${CMSIS_NN_ROOT}" cmsis-nn)
else()
  message(FATAL_ERROR "cmsis-nn CMakeLists.txt not found at: ${CMSIS_NN_ROOT}")
endif()

# ---------- Unity (tiny static lib) ----------
include(FetchContent)
FetchContent_Declare(unity GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git GIT_TAG v2.6.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(unity)
add_library(unity_fetch STATIC ${unity_SOURCE_DIR}/src/unity.c)
 target_include_directories(unity_fetch PUBLIC ${unity_SOURCE_DIR}/src)

# ---------- FVP retarget + startup ----------
add_library(retarget STATIC ${FVP_CORSTONE_300_PATH}/retarget.c ${FVP_CORSTONE_300_PATH}/uart.c)
target_compile_definitions(retarget PUBLIC USING_FVP_CORSTONE_300)
target_compile_definitions(retarget PRIVATE ENABLE_RETARGET_SYSCALLS)

# Map cortex-m* to ARMCM* for CMSIS startup
string(REGEX REPLACE "^cortex-m([0-9]+)$" "ARMCM\\1" ARM_CPU ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m33")
  set(ARM_FEATURES "_DSP_FP")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m4")
  set(ARM_FEATURES "_FP")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m7")
  set(ARM_FEATURES "_DP")
else()
  set(ARM_FEATURES "")
endif()

add_library(cmsis_startup STATIC)
 target_sources(cmsis_startup PRIVATE
   ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/startup_${ARM_CPU}.c
   ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/system_${ARM_CPU}.c)
 target_include_directories(cmsis_startup PUBLIC
   ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include
   ${CMSIS_PATH}/CMSIS/Core/Include)
 target_compile_options(cmsis_startup INTERFACE -include${ARM_CPU}${ARM_FEATURES}.h)
 target_compile_definitions(cmsis_startup PRIVATE ${ARM_CPU}${ARM_FEATURES})

# Linker script (GNU)
set(LINK_FILE ${FVP_CORSTONE_300_PATH}/linker.ld)

# ---------- Helpers ----------
function(_normalize_target_name out_var in_path)
  string(REGEX REPLACE "^artifacts/generated_tests/([^/]+).*" "\\1" _tgt "${in_path}")
  set(${out_var} "${_tgt}" PARENT_SCOPE)
endfunction()

# ---------- Discover & build tests ----------
# Discover test directories by looking for generated C files (e.g., *_conv2d.c, *_pooling.c, etc.)
file(GLOB_RECURSE GENERATED_TEST_SOURCES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "artifacts/generated_tests/*/*_conv2d.c"
  "artifacts/generated_tests/*/*_pooling.c"
  "artifacts/generated_tests/*/*_fully_connected.c"
  "artifacts/generated_tests/*/*_depthwise_conv2d.c"
  "artifacts/generated_tests/*/*_dwconv.c"
  "artifacts/generated_tests/*/*_transpose_conv.c"
  "artifacts/generated_tests/*/*_add.c"
  "artifacts/generated_tests/*/*_mul.c"
  "artifacts/generated_tests/*/*_sub.c"
  "artifacts/generated_tests/*/*_minmax.c"
  "artifacts/generated_tests/*/*_relu.c"
  "artifacts/generated_tests/*/*_relu6.c"
  "artifacts/generated_tests/*/*_leakyrelu.c"
  "artifacts/generated_tests/*/*_hardswish.c"
  "artifacts/generated_tests/*/*_prelu.c"
  "artifacts/generated_tests/*/*_tanh.c"
  "artifacts/generated_tests/*/*_logistic.c"
  "artifacts/generated_tests/*/*_softmax.c"
  "artifacts/generated_tests/*/*_reshape.c"
  "artifacts/generated_tests/*/*_transpose.c"
  "artifacts/generated_tests/*/*_pad.c"
  "artifacts/generated_tests/*/*_stridedslice.c"
  "artifacts/generated_tests/*/*_concatenation.c"
  "artifacts/generated_tests/*/*_split.c"
  "artifacts/generated_tests/*/*_argmax.c"
  "artifacts/generated_tests/*/*_argmin.c"
  "artifacts/generated_tests/*/*_reducemax.c"
  "artifacts/generated_tests/*/*_reducemin.c"
  "artifacts/generated_tests/*/*_mean.c"
  "artifacts/generated_tests/*/*_quantize.c"
  "artifacts/generated_tests/*/*_dequantize.c"
  "artifacts/generated_tests/*/*_matmul_batch.c"
  "artifacts/generated_tests/*/*_squeeze.c")

# Also discover by CMakeLists.txt in test directories
file(GLOB_RECURSE GENERATED_TEST_CMAKE
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "artifacts/generated_tests/*/CMakeLists.txt")

# Combine both discovery methods - use C files as primary, CMakeLists.txt as fallback
set(GENERATED_TEST_DIRS)
foreach(SRC_REL ${GENERATED_TEST_SOURCES})
  get_filename_component(TEST_DIR_REL "${SRC_REL}" DIRECTORY)
  list(APPEND GENERATED_TEST_DIRS "${TEST_DIR_REL}")
endforeach()
foreach(CMAKE_REL ${GENERATED_TEST_CMAKE})
  get_filename_component(TEST_DIR_REL "${CMAKE_REL}" DIRECTORY)
  if(NOT "${TEST_DIR_REL}" IN_LIST GENERATED_TEST_DIRS)
    list(APPEND GENERATED_TEST_DIRS "${TEST_DIR_REL}")
  endif()
endforeach()

# Remove duplicates
list(REMOVE_DUPLICATES GENERATED_TEST_DIRS)

if(NOT GENERATED_TEST_DIRS)
  message(WARNING "No generated_tests directories with C files found. Did you run the generator?")
endif()

set(GENERATED_TEST_TARGETS)
foreach(TEST_DIR_REL ${GENERATED_TEST_DIRS})
  get_filename_component(TEST_DIR_ABS "${TEST_DIR_REL}" ABSOLUTE)
  _normalize_target_name(TGT_NAME "${TEST_DIR_REL}")

  # Find generated C files in this test directory (e.g., basic_s8_conv2d.c)
  file(GLOB TEST_SRCS "${TEST_DIR_ABS}/*_*.c")
  # Filter out any unwanted files if needed
  list(FILTER TEST_SRCS EXCLUDE REGEX ".*test_runner\\.c$")
  
  if(TEST_SRCS STREQUAL "")
    message(WARNING "No generated C sources found in ${TEST_DIR_REL}; skipping")
    continue()
  endif()
  
  set(SRCS ${TEST_SRCS})

  add_executable(${TGT_NAME} ${SRCS})
  set_target_properties(${TGT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

  # Includes (generated API + local cmsis.h + neuralSPOT bits used by tests)
  file(GLOB NEURALSPOT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/neuralspot/ns-*/includes")
  target_include_directories(${TGT_NAME} PRIVATE
    "${TEST_DIR_ABS}/includes"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${NEURALSPOT_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/utils"
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/boards/apollo510_evb/bsp"
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/CMSIS/ARM/Include"
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/AmbiqSuite/R5.1.0_rc27/CMSIS/AmbiqMicro/Include"
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/neuralSPOT/extern/CMSIS/CMSIS-DSP-1.16.2/Include")

  target_compile_definitions(${TGT_NAME} PUBLIC USING_FVP_CORSTONE_300)
  target_compile_options(${TGT_NAME} PRIVATE -Wno-switch-default -Wno-unused-parameter -Wno-cast-align)

  # Linker script + libs
  target_link_options(${TGT_NAME} PRIVATE -T ${LINK_FILE})
  target_link_libraries(${TGT_NAME} PRIVATE unity_fetch cmsis-nn retarget cmsis_startup)

  list(APPEND GENERATED_TEST_TARGETS ${TGT_NAME})
endforeach()

if(GENERATED_TEST_TARGETS)
  add_custom_target(generated_tests ALL DEPENDS ${GENERATED_TEST_TARGETS})
endif()

# ---------- Summary ----------
message(STATUS "  CMSIS_PATH             : ${CMSIS_PATH}")
message(STATUS "  CMSIS_NN_ROOT          : ${CMSIS_NN_ROOT}")
message(STATUS "  FVP_CORSTONE_300_PATH  : ${FVP_CORSTONE_300_PATH}")
message(STATUS "  TARGET CPU             : ${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "  Coverage               : ${ENABLE_COVERAGE}")
message(STATUS "  Tests dir              : ${CMAKE_BINARY_DIR}/tests")
