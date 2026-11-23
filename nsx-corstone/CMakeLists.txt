# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
cmake_minimum_required(VERSION 3.19)

if(NOT TARGET nsx::corstone_cpu)
    message(FATAL_ERROR "nsx-corstone-cmsis-nn requires nsx-corstone-soc >= 0.2.0 (nsx::corstone_cpu)")
endif()
if(DEFINED NSX_CMSIS_NN_LIB OR ARM_NN_ENABLE_F32 OR ARM_NN_ENABLE_F16)
    message(FATAL_ERROR "nsx-corstone-cmsis-nn supports source-built integer kernels; unset NSX_CMSIS_NN_LIB and disable ARM_NN_ENABLE_F32/F16")
endif()

# Reuse the NSX library target with the Corstone provider's CPU requirements.
set(NSX_BOARD_FLAGS_TARGET nsx::corstone_cpu)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../nsx" cmsis-nn)
