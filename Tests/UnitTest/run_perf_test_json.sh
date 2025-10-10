#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Version: 1.0
# Date: 2025-07-23
# This bash script downloads unit test dependencies, builds unit tests and also runs the tests.

CPU="cortex-m55"
QUIET=0
USE_ARM_COMPILER=0
USE_FVP_FROM_DOWNLOAD=1
USE_PYTHON_VENV=1

usage="
Helper script to setup and run NS-CMSIS-NN performance tests

args:
    -h  Display this message.
    -c  Target cpu. Takes multiple arguments as a comma seperated list. eg cortex-m3,cortex-m7,cortex-m55 (default: cortex-m55)
    -q  Quiet mode. This reduces the amount of info printed from building and running cmsis-unit tests.
    -e  Disable environment setup. This flag will stop the script from attempting to download dependencies. This is just a quiet mode to reduce print outs.
    -a  Use Arm Compiler that is previously available on machine. Ensure compiler directory is added to path and export CC.
    -f  Disable the usage of FVP from download directory. Requires FVP to be in path before calling script.
    example usage: $(basename "$0") -c cortex-m3,cortex-m4 -o '-O2' -q
"

while getopts hc:qeaf: flag
do
    case "${flag}" in
        h) echo "${usage}"
           exit 1;;
        c) CPU=${OPTARG};;
        q) QUIET=1;;
        e) SETUP_ENVIRONMENT=0;;
        a) USE_ARM_COMPILER=0;;
        f) USE_FVP_FROM_DOWNLOAD=0;;
    esac
done

Setup_Environment() {
    if [[ ! -d ${WORKING_DIR}/corstone300_download || ! -d ${WORKING_DIR}/CMSIS_5 || ! -d ${WORKING_DIR}/ethos-u-core-platform || ! -d ${WORKING_DIR}/cmsis_nn_venv ]]; then
        pushd ..
        ./build_and_run_tests.sh -c ${CPU} -b -r
        popd
    fi
}

Run_Perf_Tests() {
    echo "++ Running Performance Tests"

    ./build_and_run_tests.sh -c ${CPU} -r
    

    if [[ -f ./perf_results_${cpu}.json ]]; then
        rm -f ./perf_results_${cpu}.json
    fi

    IFS=$'\n' tests=($(find ./build-${cpu}-${compiler}/ -iname "*.elf" | grep ".*perf.*"))
    
    echo "[" > ./perf_results_${cpu}.json
    for test in "${tests[@]}"
    do
        echo "Test: ${test}"
        FVP_Corstone_SSE-300_Ethos-U55 -C mps3_board.uart0.shutdown_on_eot=1 -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file="-" -C mps3_board.uart0.unbuffered_output=1 ${test} | grep "{.*}" >> ./perf_results_${cpu}.json
    done

    echo "{\"date\":\"$(date '+%b %d %Y %H:%M:%S')\", \"op\": \"dummy\", \"shape\": \"none\", \"DWT_CYCCNT\":0, \"PMU\":{\"ARM_PMU_INST_RETIRED\":0, \"ARM_PMU_MVE_FP_RETIRED\":0, \"ARM_PMU_MVE_INST_RETIRED\":0, \"ARM_PMU_MVE_INT_MAC_RETIRED\":0}}" >> ./perf_results_${cpu}.json
    echo "]" >> ./perf_results_${cpu}.json

    if [[ ! -f ./perf_results_${cpu}_ref.json ]]; then
        cp ./perf_results_${cpu}.json ./perf_results_${cpu}_ref.json
    fi

    output=$(./parse_and_comp_perf_results.py --perf_test_results_json=./perf_results_${cpu}.json --perf_golden_results_json=./perf_results_${cpu}_ref.json)
    echo "$output" | grep "No Failures" -vqz

    if [[ $? -eq 0 ]]; then
        echo "${output}"
        exit 1
    else
        echo "${output}"
        echo "Performance Tests for ${cpu} ran successfully."
        exit 0
    fi
}

UNAME_M=$(uname -m)
UNAME_S=$(uname -s)


if [[ ${UNAME_S} != Linux ]]; then
    echo "Error: This script only supports Linux."
    exit 1
fi


mkdir -p downloads
pushd downloads
WORKING_DIR=$(pwd)

echo "++ Setting Environment"
Setup_Environment

if [[ ${USE_PYTHON_VENV} -eq 1 ]]; then
    source ./cmsis_nn_venv/bin/activate
fi

if [[ -z "${ETHOS_U_CORE_PLATFORM_PATH}" ]]; then
    ETHOS_U_CORE_PLATFORM_PATH="${WORKING_DIR}/ethos-u-core-platform"
fi

if [[ -z "${CMSIS_5_PATH}" ]]; then
    CMSIS_5_PATH="${WORKING_DIR}/CMSIS_5"
fi

popd
IFS=',' read -r -a cpu_array <<< "$CPU"

for cpu in "${cpu_array[@]}"
do
    echo "++ Targetting ${cpu}"

    if [[ ${USE_ARM_COMPILER} -eq 1 ]]; then
        compiler="arm-compiler"
    else
        compiler="gcc"
    fi

    if [[ $USE_FVP_FROM_DOWNLOAD -eq 1 ]]; then
        if [[ ${UNAME_M} == x86_64 ]]; then
            export PATH=${WORKING_DIR}/corstone300_download/models/Linux64_GCC-9.3/:${PATH}
        elif [[ ${UNAME_M} == aarch64 ]]; then
            export PATH=${WORKING_DIR}/corstone300_download/models/Linux64_armv8l_GCC-9.3/:${PATH}
        fi
    fi

    Run_Perf_Tests
done

echo ""
echo "++ Tests for ${CPU} ran successfully"
if [[ ${USE_PYTHON_VENV} -eq 1 ]]; then
    deactivate
fi
