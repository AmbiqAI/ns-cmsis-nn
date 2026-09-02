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
#
# Keep each *_COMMIT line spelled exactly NAME="<40 hex>", with no trailing
# comment and no trailing whitespace: unity-m55-compile.yml's assertion step
# reads these lines with an anchored sed and fails the job outright if it
# cannot read exactly one match. Put the branch/date note on its own line
# above, as below.
#
# Three flags bypass these pins, all of them deliberately: -e skips
# Setup_Environment altogether (so nothing is cloned, fetched or checked),
# while -C and -u point the build at a CMSIS_5 or ethos-u-core-platform tree
# you supply, which this script neither clones nor verifies. CI passes none
# of the three.
#
# Bump ownership: CMSIS_5's upstream is archived, so its pin is effectively
# permanent and only moves if this harness moves off CMSIS_5 entirely.
# ethos-u-core-platform lives on gitlab.arm.com, which no bot in this repo
# watches, so its pin is bumped by hand; validate a bump with the PR compile
# gate plus a manual legacy-tester dispatch, which is the only leg that runs
# the built ELFs under the FVP.
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

# git in <dir>, tolerating a checkout git considers to be owned by someone
# else: CI restores this tree from a cache rather than checking it out, and a
# dev container can run as a different uid than the one that first cloned --
# either trips git's dubious-ownership guard. Same `-c safe.directory` the
# assertion step in unity-m55-compile.yml uses, for the same reason.
Git_At() {
    local dir="$1"
    shift
    git -c safe.directory='*' -C "${dir}" "$@"
}

# Print the HEAD commit of <dir>, or nothing if it is not a git checkout.
Head_Of() {
    Git_At "$1" rev-parse HEAD 2>/dev/null || true
}

# Fetch <commit> into the existing git repository at <dir> from its origin.
# The commit is fetched directly at depth 1, which both remotes pinned above
# serve today; the fallback is for a server that will not serve a bare commit
# that way.
#
# The fallback has to *deepen*, not just fetch. Every tree on this path is
# shallow: Clone_At_Commit fetches at depth 1, so the repository carries a
# shallow boundary. A plain `git fetch origin` there does pick up commits
# *newer* than that boundary, but it can never reach anything older -- so a
# pin that is not a descendant of the boundary (a downgrade after a bad bump,
# or a force-push upstream) comes back "fetched successfully" with the pinned
# object still absent, and the caller then reports a perfectly good pin as
# "the remote has no such commit". Measured both ways in the container before
# this line was written. --unshallow is what removes the boundary; it errors
# on a repository that is already complete, hence the second form. --tags on
# both, so a pin expressed as a tag object in future is fetched rather than
# silently missing.
Fetch_Commit() {
    local name="$1" dir="$2" commit="$3"

    if Git_At "${dir}" fetch --quiet --depth=1 origin "${commit}"; then
        return 0
    fi
    echo "   ${name}: shallow fetch of that commit failed, fetching in full"
    Git_At "${dir}" fetch --quiet --tags --unshallow origin \
        || Git_At "${dir}" fetch --quiet --tags origin \
        || return 1
}

# Print the paths that make <dir> differ from its own HEAD: modified tracked
# files and untracked files, but not files the upstream repository's own
# .gitignore covers. Empty output means the working tree is exactly the
# checked-out commit, which is the half of "this tree is the pinned tree" that
# comparing HEADs cannot see.
Dirty_Paths() {
    Git_At "$1" status --porcelain --ignored=no 2>/dev/null || true
}

# Check <dir> out at <commit>, separating the two failure classes Retry treats
# differently. If the object is not there after a successful fetch, the remote
# does not have that commit: a wrong pin, which no number of retries fixes
# (125). If the object *is* there and the checkout still fails, the fault is
# local -- a full disk, a corrupt pack, an index lock left by a killed git --
# and another attempt may well clear it, so it keeps its retry budget (1).
Checkout_Commit() {
    local name="$1" url="$2" dir="$3" commit="$4"

    if ! Git_At "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        echo "   ${name}: fetched ${url}, but it has no commit ${commit}." >&2
        echo "   Check the pin at the top of this script." >&2
        return 125
    fi
    if ! Git_At "${dir}" checkout --quiet "${commit}"; then
        echo "   ${name}: ${commit} is present in ${dir} but will not check out." >&2
        echo "   That is a local fault, not a bad pin -- retrying may clear it." >&2
        return 1
    fi
}

# Clone <url> into <dir> at exactly <commit>. Any tree left behind by an
# earlier attempt is removed first, so a retry never resumes into a
# half-written clone; that is the same hazard the "already installed" checks
# in Setup_Environment cannot see, and the Assert_Pinned call after each clone
# is what catches it if it ever survives this far.
Clone_At_Commit() {
    local name="$1" url="$2" dir="$3" commit="$4"

    rm -rf "${dir}"
    git -c init.defaultBranch=main init --quiet "${dir}" || return 1
    Git_At "${dir}" remote add origin "${url}" || return 1
    Fetch_Commit "${name}" "${dir}" "${commit}" || return 1
    Checkout_Commit "${name}" "${url}" "${dir}" "${commit}"
}

# Move an existing checkout to <commit> without re-cloning it: point origin at
# the pinned URL if it is not already, then fetch and check out. Retried by
# the caller, so network faults get the same budget a fresh clone gets.
Move_To_Commit() {
    local name="$1" url="$2" dir="$3" commit="$4"
    local origin_url

    origin_url="$(Git_At "${dir}" remote get-url origin 2>/dev/null || true)"
    if [[ "${origin_url}" != "${url}" ]]; then
        echo "   ${name}: origin is ${origin_url:-unset}, repointing it at ${url}"
        # Both of these are retryable (1), not fatal (125): the usual reason
        # either fails is a .git/config.lock left by a killed git, which the
        # next attempt often finds gone. Nothing here is swallowed, so git's
        # own diagnosis of the failure reaches the log -- swallowing the
        # remove is what used to turn a stale lock into a confusing "remote
        # origin already exists" one line later.
        if [[ -n "${origin_url}" ]]; then
            Git_At "${dir}" remote remove origin || return 1
        fi
        Git_At "${dir}" remote add origin "${url}" || return 1
    fi
    Fetch_Commit "${name}" "${dir}" "${commit}" || return 1
    Checkout_Commit "${name}" "${url}" "${dir}" "${commit}"
}

# Leave <dir> sitting exactly on <commit>, with a working tree that is that
# commit, or exit. Runs on every path, not only after a fresh clone: a
# directory restored from a CI cache, or left behind by an older unpinned
# checkout, is skipped by the "already installed" branches in
# Setup_Environment and would otherwise be used at whatever commit -- and
# whatever local edits -- it happens to hold.
#
# Two conditions, not one. HEAD must equal the pin, and the working tree must
# be clean, because `git checkout` carries divergence forward: a local edit to
# a file that happens not to differ between the old commit and the pin
# survives a move to the pin untouched, and HEAD alone reports that tree as
# pinned. A dirty tree is refused with exit 125 -- this script's do-not-retry
# status -- and never repaired: `reset --hard` or `clean -fd` here would
# silently destroy work whose only copy is that directory.
#
# An existing *clean* checkout at the wrong commit is moved to the pin rather
# than rejected. This is a developer-machine and dev-container path, not a CI
# one: CI has no restore-keys, the cache key hashes this file, so a pin bump
# misses the cache and takes the fresh-clone path instead. What it fixes is
# .devcontainer/install.sh, which runs this harness, and any local checkout
# made before a pin moved -- "delete the directory and re-run" is a poor
# remedy for a clone the developer did not know was there. A checkout that is
# not a git repository at all is still a hard failure: something the harness
# did not create is sitting where the clone belongs, and guessing is worse
# than stopping.
Assert_Pinned() {
    local name="$1" url="$2" dir="$3" commit="$4"
    local head was branch dirty

    head="$(Head_Of "${dir}")"
    if [[ -z "${head}" ]]; then
        echo "${name}: ${dir} exists but is not a git checkout." >&2
        echo "  repository:  ${url}" >&2
        echo "  pinned at:   ${commit}" >&2
        echo "  Delete ${dir} and re-run to re-clone at the pinned commit." >&2
        exit 1
    fi

    dirty="$(Dirty_Paths "${dir}")"
    if [[ -n "${dirty}" ]]; then
        echo "${name}: working tree has local changes, so it is not the" >&2
        echo "pinned tree whatever HEAD says." >&2
        echo "  repository:  ${url}" >&2
        echo "  pinned at:   ${commit}" >&2
        echo "  HEAD is:     ${head}" >&2
        echo "  changed paths (git status --porcelain):" >&2
        printf '%s\n' "${dirty}" | sed 's|^|    |' >&2
        echo "  This script will not discard them: commit, stash or delete" >&2
        echo "  them yourself, or leave them where they are and point the" >&2
        echo "  build at another tree with -C (CMSIS_5) or -u" >&2
        echo "  (ethos-u-core-platform). Re-running as-is will not help." >&2
        exit 125
    fi

    if [[ "${head}" == "${commit}" ]]; then
        echo "   ${name} pinned at ${commit} -- ok."
        return 0
    fi

    was="${head}"
    branch="$(Git_At "${dir}" symbolic-ref -q --short HEAD 2>/dev/null || true)"
    echo "   ${name}: HEAD is ${was}${branch:+ (branch ${branch})}, pinned at ${commit} -- moving it to the pin."
    if Retry "move ${name} to its pin" \
        Move_To_Commit "${name}" "${url}" "${dir}" "${commit}"; then
        head="$(Head_Of "${dir}")"
    fi

    if [[ "${head}" != "${commit}" ]]; then
        echo "${name}: clone is not at its pin and could not be moved to it." >&2
        echo "  repository:  ${url}" >&2
        echo "  pinned at:   ${commit}" >&2
        echo "  HEAD is:     ${head}" >&2
        echo "  Delete ${dir} and re-run to re-clone at the pinned commit." >&2
        exit 1
    fi
    echo "   ${name} pinned at ${commit} -- ok (moved from ${was}${branch:+, which was branch ${branch}})."
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
