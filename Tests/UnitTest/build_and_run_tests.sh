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
# Date: 2024-05-21
# This bash script downloads unit test dependencies, builds unit tests and also runs the tests.

CPU="cortex-m55"
OPTIMIZATION="-Ofast"
QUIET=0
BUILD=1
RUN=1
SETUP_ENVIRONMENT=1
USE_ARM_COMPILER=0
USE_PYTHON_VENV=1
USE_FVP_FROM_DOWNLOAD=1
USE_GCC_FROM_DOWNLOAD=1

ETHOS_U_CORE_PLATFORM_PATH=""
CMSIS_5_PATH=""
CMAKE_EXTRA_DEFS=""

# Third-party clone pins (AmbiqAI/ns-cmsis-nn#402). Both clones below used to
# track their upstream default branch, so an upstream push could change -- or
# break -- this repo's PR-required compile gate with no change in this repo,
# and a transient network failure failed that gate outright. The pins are
# recorded here, in one place, alongside the toolchain pin that GCC_URL
# carries in Setup_Environment below; CI asserts the checked-out HEADs against
# these same variables.
#
# To bump: put the new full 40-character commit here (the HEAD check compares
# the whole string, so an abbreviation will not do) and update the trailing
# branch/date comment. Nothing else needs touching -- the CI downloads cache
# keys on a hash of this file, so a bump misses the cache and re-clones
# instead of reusing the previously pinned tree.
CMSIS_5_URL="https://github.com/ARM-software/CMSIS_5.git"
# develop @ 2024-09-03; the upstream repository is archived.
CMSIS_5_COMMIT="55b19837f5703e418ca37894d5745b1dc05e4c91"
ETHOS_U_CORE_PLATFORM_URL="https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform"
# main @ 2026-08-19.
ETHOS_U_CORE_PLATFORM_COMMIT="cec1a0ae3f05b2cf9a1518c7087cda96aed322a0"

usage="
Helper script to setup, build and run CMSIS-NN unit tests

args:
    -h  Display this message.
    -c  Target cpu. Takes multiple arguments as a comma seperated list. eg cortex-m3,cortex-m7,cortex-m55 (default: cortex-m55)
    -o  Optimization level. (default: '-Ofast')
    -q  Quiet mode. This reduces the amount of info printed from building and running cmsis-unit tests.
    -b  Disable CMake build. Only works with previously built targets. Designed to quickly rerun cpu targets.
    -r  Disable running the unit tests. Designed to test build only or allow user to manually run individual test cases outside of this script.
    -e  Disable environment setup. This flag will stop the script from attempting to download dependencies. This is just a quiet mode to reduce print outs.
    -a  Use Arm Compiler that is previously available on machine. Ensure compiler directory is added to path and export CC.
    -p  Disable the usage of python venv from download directory. Requires dependencies to be install before calling script.
    -f  Disable the usage of FVP from download directory. Requires FVP to be in path before calling script.
    -u  Path to ethos-u-core-platform
    -g  Disable the usage of GCC that is already from download directory. Requires gcc to be in path before calling script.
    -C  Path to cmsis 5
    -D  Pass through a CMake -D argument, e.g. CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=ON (default: empty)

    example usage: $(basename "$0") -c cortex-m3,cortex-m4 -o '-O2' -q
"

while getopts hc:o:qbreapfu:gC:D: flag
do
    case "${flag}" in
        h) echo "${usage}"
           exit 1;;
        c) CPU=${OPTARG};;
        o) OPTIMIZATION=${OPTARG};;
        q) QUIET=1;;
        b) BUILD=0;;
        r) RUN=0;;
        e) SETUP_ENVIRONMENT=0;;
        a) USE_ARM_COMPILER=1;;
        p) USE_PYTHON_VENV=0;;
        f) USE_FVP_FROM_DOWNLOAD=0;;
        u) ETHOS_U_CORE_PLATFORM_PATH="${OPTARG}";;
        g) USE_GCC_FROM_DOWNLOAD=0;;
        C) CMSIS_5_PATH="${OPTARG}";;
        D) CMAKE_EXTRA_DEFS="${CMAKE_EXTRA_DEFS} -D${OPTARG}";;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)

# Bounded retry around one network step (#402): three attempts with a widening
# backoff and a log line per attempt, so a transient GitHub or GitLab failure
# does not fail a required check on its own. The command is run as given, so
# both plain commands and the shell functions below can be passed to it.
Retry() {
    local what="$1"
    shift
    local attempt=1
    local max_attempts=3
    local delay=5

    while true; do
        echo "   ${what}: attempt ${attempt} of ${max_attempts}"
        local status=0
        "$@" || status=$?
        if [[ ${status} -eq 0 ]]; then
            return 0
        fi
        # 125 is this script's "do not retry" signal: the step failed for a
        # reason another attempt cannot fix, such as a pinned commit the
        # remote does not have. Retrying that would only re-download the
        # repository twice more.
        if [[ ${status} -eq 125 ]]; then
            echo "   ${what}: giving up after attempt ${attempt}, retrying cannot fix this." >&2
            return 1
        fi
        if [[ ${attempt} -ge ${max_attempts} ]]; then
            echo "   ${what}: failed after ${max_attempts} attempts." >&2
            return 1
        fi
        echo "   ${what}: attempt ${attempt} failed, retrying in ${delay}s." >&2
        sleep "${delay}"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

# Clone <url> into <dir> at exactly <commit>. The commit is fetched directly
# at depth 1, which both remotes pinned above serve today; a full fetch is the
# fallback for a server that will not serve a bare commit that way, and it is
# also what turns a wrong pin into a clear "no such commit". Any tree
# left behind by an earlier attempt is removed first, so a retry never resumes
# into a half-written clone; that is the same hazard the "already installed"
# checks in Setup_Environment cannot see, and the Assert_Pinned call after
# each clone is what catches it if it ever survives this far.
Clone_At_Commit() {
    local name="$1" url="$2" dir="$3" commit="$4"

    rm -rf "${dir}"
    git -c init.defaultBranch=main init --quiet "${dir}" || return 1
    git -C "${dir}" remote add origin "${url}" || return 1
    if ! git -C "${dir}" fetch --quiet --depth=1 origin "${commit}"; then
        echo "   ${name}: shallow fetch of that commit failed, trying a full fetch"
        git -C "${dir}" fetch --quiet origin || return 1
    fi
    if ! git -C "${dir}" checkout --quiet "${commit}"; then
        echo "   ${name}: fetched ${url}, but it has no commit ${commit}." >&2
        echo "   Check the pin at the top of this script." >&2
        return 125
    fi
}

# Fail loud unless <dir> sits exactly on <commit>. Runs on every path, not
# only after a fresh clone: a directory restored from a CI cache, or left
# behind by an older unpinned checkout, is skipped by the "already installed"
# branches above and would otherwise be used at whatever commit it happens to
# hold.
Assert_Pinned() {
    local name="$1" url="$2" dir="$3" commit="$4"
    local head

    head="$(git -C "${dir}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${head}" != "${commit}" ]]; then
        echo "${name}: clone is not at its pin." >&2
        echo "  repository:  ${url}" >&2
        echo "  pinned at:   ${commit}" >&2
        echo "  HEAD is:     ${head:-(none - ${dir} is not a git checkout)}" >&2
        echo "  Delete ${dir} and re-run to re-clone at the pinned commit." >&2
        exit 1
    fi
    echo "   ${name} pinned at ${commit} -- ok."
}

Setup_Environment() {
    set -e
    echo "++ Downloading Corstone300"
    if [[ -d ${WORKING_DIR}/corstone300_download ]]; then
        echo "Corstone300 already installed. If you wish to install a new version, please delete the old folder."
    else
        if [[ ${UNAME_M} == x86_64 ]]; then
            CORSTONE_URL=https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz
        elif [[ ${UNAME_M} == aarch64 ]]; then
            CORSTONE_URL=https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64_armv8l.tgz
        fi

        TEMPFILE=$(mktemp -d)/temp_file
        wget -q "${CORSTONE_URL}" -O "${TEMPFILE}" || {
            echo "Download Corstone300 failed!" >&2
            exit 1
        }

        TEMPDIR=$(mktemp -d)
        tar -C ${TEMPDIR} -xzf ${TEMPFILE} >&2
        mkdir ${WORKING_DIR}/corstone300_download
        ${TEMPDIR}/FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --no-interactive -q -d ${WORKING_DIR}/corstone300_download >&2
    fi

    echo "++ Downloading GCC"
    if [[ -d ${WORKING_DIR}/arm_gcc_download ]]; then
        echo "Arm GCC already installed. If you wish to install a new version, please delete the old folder."
    else
        if [[ ${UNAME_M} == x86_64 ]]; then
            GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz"
        elif [[ ${UNAME_M} == aarch64 ]]; then
            GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-aarch64-arm-none-eabi.tar.xz"
        fi

        TEMPFILE=$(mktemp -d)/temp_file
        wget -q ${GCC_URL} -O ${TEMPFILE} || {
            echo "Download Arm GCC failed!" >&2
            exit 1
        }
        mkdir ${WORKING_DIR}/arm_gcc_download

        tar -C ${WORKING_DIR}/arm_gcc_download --strip-components=1 -xJf ${TEMPFILE} >&2
    fi

    echo "++ Cloning CMSIS-5"
    if [[ -d ${WORKING_DIR}/CMSIS_5 ]]; then
        echo "CMSIS-5 already installed. If you wish to install a new version, please delete the old folder."
    else
        Retry "clone CMSIS_5" Clone_At_Commit \
            "CMSIS_5" "${CMSIS_5_URL}" "${WORKING_DIR}/CMSIS_5" "${CMSIS_5_COMMIT}"
    fi
    Assert_Pinned "CMSIS_5" "${CMSIS_5_URL}" "${WORKING_DIR}/CMSIS_5" "${CMSIS_5_COMMIT}"

    echo "++ Cloning Ethos-U core platform"
    if [[ -d ${WORKING_DIR}/ethos-u-core-platform ]]; then
        echo "Ethos-U core platform already installed. If you wish to install a new version, please delete the old folder."
    else
        Retry "clone ethos-u-core-platform" Clone_At_Commit \
            "ethos-u-core-platform" "${ETHOS_U_CORE_PLATFORM_URL}" \
            "${WORKING_DIR}/ethos-u-core-platform" "${ETHOS_U_CORE_PLATFORM_COMMIT}"
    fi
    Assert_Pinned "ethos-u-core-platform" "${ETHOS_U_CORE_PLATFORM_URL}" \
        "${WORKING_DIR}/ethos-u-core-platform" "${ETHOS_U_CORE_PLATFORM_COMMIT}"

    echo "++ Setting up python environment"
    if [[ -d ${WORKING_DIR}/cmsis_nn_venv ]]; then
        echo "Python venv already installed. If you wish to install a new version, please delete the old folder."
    else
        python3 -m venv cmsis_nn_venv
        source cmsis_nn_venv/bin/activate
        pip3 install --disable-pip-version-check -qq -r ../requirements.txt
        deactivate
    fi
}

Build_Tests() {
    set -e
    echo "++ Building Tests"

    if [[ ${QUIET} -eq 0 ]]; then

        cmake -S ./ -B build-${cpu}-${compiler} \
            -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
            -DTARGET_CPU=${cpu} \
            -DCMSIS_PATH=${CMSIS_5_PATH} \
            -DCMSIS_OPTIMIZATION_LEVEL=${OPTIMIZATION} \
            ${CMAKE_EXTRA_DEFS}
        cmake --build build-${cpu}-${compiler}/

        echo "Built successfully into build-${cpu}-${compiler}"
    else

        mkdir -p build-${cpu}-${compiler}

        cmake -S ./ -B build-${cpu}-${compiler} \
            -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
            -DTARGET_CPU=${cpu} \
            -DCMSIS_PATH=${CMSIS_5_PATH} \
            -DCMSIS_OPTIMIZATION_LEVEL=${OPTIMIZATION} \
            ${CMAKE_EXTRA_DEFS} \
          > build-${cpu}-${compiler}/cmake_stdout.log

        cmake --build build-${cpu}-${compiler}/ \
          > build-${cpu}-${compiler}/make_stdout.log

        echo "Built successfully into build-${cpu}-${compiler}"
    fi
}

Run_Tests() {
    set +e
    echo "++ Running Unit Tests"
    readarray -d '' tests < <(find ./build-${cpu}-${compiler}/ -iname "*.elf" -print0)
    for test in "${tests[@]}"
    do
        echo "Test: ${test}"
        output=$(FVP_Corstone_SSE-300_Ethos-U55 -C mps3_board.uart0.shutdown_on_eot=1 -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file="-" -C mps3_board.uart0.unbuffered_output=1 ${test})
        echo "$output" | grep "0 Failures" -vqz
        if [[ $? -eq 0 ]]; then
            echo "${output}"
            echo "${test} failed. Script exiting."
            exit 1
        elif [[ ${QUIET} -eq 0 ]]; then
            echo "${output}"
        fi
    done
    echo "Tests for ${cpu} ran successfully."
}

if [[ ${BUILD} -eq 0 && ${RUN} -eq 0 && ${SETUP_ENVIRONMENT} -eq 0 ]]; then
    echo "All script functions are disabled. Script will exit and do nothing."
    exit 1
fi

UNAME_M=$(uname -m)
UNAME_S=$(uname -s)

if [[ ${UNAME_S} != Linux ]]; then
    echo "Error: This script only supports Linux."
    exit 1
fi

mkdir -p downloads
pushd downloads
cd $(pwd -P)
WORKING_DIR=$(pwd)

if [[ ${SETUP_ENVIRONMENT} -eq 1 ]]; then
    echo "++ Setting Environment"
    Setup_Environment
fi

if [[ ${USE_PYTHON_VENV} -eq 1 ]]; then
    source cmsis_nn_venv/bin/activate
fi

if [[ -z "${ETHOS_U_CORE_PLATFORM_PATH}" ]]; then
    ETHOS_U_CORE_PLATFORM_PATH="${WORKING_DIR}/ethos-u-core-platform"
fi

if [[ -z "${CMSIS_5_PATH}" ]]; then
    CMSIS_5_PATH="${WORKING_DIR}/CMSIS_5"
fi

popd
IFS=',' read -r -a cpu_array <<< "$CPU"

if [[ ${BUILD} -eq 1 || ${RUN} -eq 1 ]]; then
    for cpu in "${cpu_array[@]}"
    do
        echo "++ Targetting ${cpu}"
        if [[ ${USE_ARM_COMPILER} -eq 1 ]]; then
            compiler="arm-compiler"
            TOOLCHAIN_FILE=${ETHOS_U_CORE_PLATFORM_PATH}/cmake/toolchain/armclang.cmake
        else
            if [[ ${USE_GCC_FROM_DOWNLOAD} -eq 1 ]]; then
                export PATH=${WORKING_DIR}/arm_gcc_download/bin/:${PATH}
            fi
            compiler="gcc"
            TOOLCHAIN_FILE=${ETHOS_U_CORE_PLATFORM_PATH}/cmake/toolchain/arm-none-eabi-gcc.cmake
        fi

        if [[ $USE_FVP_FROM_DOWNLOAD -eq 1 ]]; then
            if [[ ${UNAME_M} == x86_64 ]]; then
                export PATH=${WORKING_DIR}/corstone300_download/models/Linux64_GCC-9.3/:${PATH}
            elif [[ ${UNAME_M} == aarch64 ]]; then
                export PATH=${WORKING_DIR}/corstone300_download/models/Linux64_armv8l_GCC-9.3/:${PATH}
            fi
        fi

        if [[ ${BUILD} -eq 1 ]]; then
            Build_Tests
        fi

        if [[ ${RUN} -eq 1 ]]; then
            Run_Tests
        fi
    done
fi

echo ""
echo "++ Tests for ${CPU} ran successfully"
if [[ ${USE_PYTHON_VENV} -eq 1 ]]; then
    deactivate
fi
