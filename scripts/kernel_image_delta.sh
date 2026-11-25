#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# The two things docs/contributing.md asks of every kernel PR ("Kernel structure
# and code size"), for the float32 fully connected and argmax entries: a function
# map of their objects, and the image delta from a --gc-sections link.
#
# For each ref, the library's Include/ and Source/ are taken with git archive,
# every Source/ file is compiled into one archive with one section per function
# and per data object, and three programs are linked against it with
# --gc-sections: one that calls arm_fully_connected_f32 only, one that calls
# arm_argmax_f32 only, and one that calls neither (the floor). The script prints
#   - the function map: `nm -S --size-sort` of the argmax, argmin, fully
#     connected and nt_t matmul objects (static inline helpers that leave no
#     symbol were inlined into the entry that lists them);
#   - each image's text, data and bss, and each later ref's delta against the
#     first;
#   - the defined function symbols each image gains or loses against the first
#     ref, so a symbol the program does not call shows up by name.
# Sizes are the linker's; nothing here runs or times the code.
#
# Usage:
#   scripts/kernel_image_delta.sh [options] <base-ref> <ref> [<ref>...]
#
# Options:
#   --cpu <cpu>    -mcpu value (default cortex-m55+nomve)
#   --opt <flags>  optimisation flags (default -O2)
#   --f16 <0|1>    ARM_NN_ENABLE_F16 (default 0; the header refuses float16
#                  without an FPU half-precision extension, e.g. on +nomve)
#   --jobs <n>     parallel compiles (default 3)
#   --keep <dir>   build under <dir> and keep it (default: a temporary
#                  directory, removed on exit)
#
# Needs git and arm-none-eabi-{gcc,ar,nm,size} on PATH.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cpu="cortex-m55+nomve"
opt="-O2"
f16="0"
jobs="3"
keep=""
refs=()

usage() {
  sed -n '/^# Usage:/,/^# Needs/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) cpu="$2"; shift 2 ;;
    --opt) opt="$2"; shift 2 ;;
    --f16) f16="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --keep) keep="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) refs+=("$1"); shift ;;
  esac
done

if [[ ${#refs[@]} -lt 2 ]]; then
  usage >&2
  exit 2
fi

for tool in git arm-none-eabi-gcc arm-none-eabi-ar arm-none-eabi-nm arm-none-eabi-size; do
  command -v "${tool}" >/dev/null || { echo "${tool} not on PATH" >&2; exit 3; }
done

if [[ -n "${keep}" ]]; then
  mkdir -p "${keep}"
  work="$(cd "${keep}" && pwd)"
else
  work="$(mktemp -d "${TMPDIR:-/tmp}/kernel_image_delta.XXXXXX")"
  trap 'rm -rf "${work}"' EXIT
fi

arch=(-mcpu="${cpu}" -mthumb -mfloat-abi=hard)
read -r -a opt_flags <<< "${opt}"
cflags=("${arch[@]}" "${opt_flags[@]}" -ffunction-sections -fdata-sections
        -DARM_NN_ENABLE_F32=1 -DARM_NN_ENABLE_F16="${f16}")
objects=(Source/BasicMathFunctions/arm_argmax_f32.c
         Source/BasicMathFunctions/arm_argmin_f32.c
         Source/FullyConnectedFunctions/arm_fully_connected_f32.c
         Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_f32.c)
programs=(floor fc argmax)

mkdir -p "${work}/programs"
cat > "${work}/programs/floor.c" <<'EOF'
volatile int image_delta_sink;
int main(void) { return image_delta_sink; }
EOF
cat > "${work}/programs/fc.c" <<'EOF'
#include "arm_nnfunctions.h"
static float32_t input[16];
static float32_t weights[64];
static float32_t bias[4];
static float32_t output[4];
volatile int32_t image_delta_format;
int main(void)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {1, 1, 1, 16};
    const cmsis_nn_dims filter_dims = {16, 1, 1, 4};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 4};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    cmsis_nn_fc_params_f32 params = {{-3.0e38f, 3.0e38f}, 0};
    params.weight_format = image_delta_format;
    return (int)arm_fully_connected_f32(&ctx, &params, &input_dims, input, &filter_dims, weights, &bias_dims, bias,
                                        &output_dims, output, ARM_NN_LAYOUT_NHWC);
}
EOF
cat > "${work}/programs/argmax.c" <<'EOF'
#include "arm_nnfunctions.h"
static float32_t input[16];
static int32_t output[4];
volatile int32_t image_delta_axis = 3;
int main(void)
{
    const cmsis_nn_dims dims = {1, 1, 4, 4};
    return (int)arm_argmax_f32(input, &dims, image_delta_axis, output);
}
EOF

compile_one() {
  local src="$1" obj
  obj="${build}/obj/$(echo "${src}" | tr / _).o"
  arm-none-eabi-gcc "${cflags[@]}" -I"${build}/src/Include" -c "${build}/src/${src}" -o "${obj}" \
    2>"${obj}.log" || { echo "FAIL ${src}"; cat "${obj}.log" >&2; }
}
export -f compile_one

declare -a labels=()
for ref in "${refs[@]}"; do
  label="$(git -C "${repo_root}" rev-parse --short=8 "${ref}^{commit}")"
  labels+=("${label}")
  build="${work}/${label}"
  rm -rf "${build}"
  mkdir -p "${build}/src" "${build}/obj" "${build}/elf"
  git -C "${repo_root}" archive "${label}" Include Source | tar -x -C "${build}/src"
  export build
  export cflags_line="${cflags[*]}"
  # The single quotes are deliberate: each worker expands its own arguments.
  # shellcheck disable=SC2016
  fails="$(cd "${build}/src" && find Source -name '*.c' | sort \
    | xargs -P "${jobs}" -n 1 bash -c 'read -r -a cflags <<< "${cflags_line}"; compile_one "$0"')"
  if [[ -n "${fails}" ]]; then
    echo "${fails}" >&2
    echo "library build failed at ${ref}" >&2
    exit 4
  fi
  arm-none-eabi-ar rcs "${build}/libheliacore.a" "${build}"/obj/*.o
  for p in "${programs[@]}"; do
    arm-none-eabi-gcc "${cflags[@]}" -I"${build}/src/Include" -c "${work}/programs/${p}.c" -o "${build}/elf/${p}.o"
    arm-none-eabi-gcc "${arch[@]}" --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections \
      "${build}/elf/${p}.o" "${build}/libheliacore.a" -o "${build}/elf/${p}.elf" >"${build}/elf/${p}.link.log" 2>&1
    arm-none-eabi-nm --defined-only "${build}/elf/${p}.elf" | awk '$2 ~ /^[Tt]$/ {print $3}' | sort \
      > "${build}/elf/${p}.functions"
  done
done

echo "kernel image delta: ${cpu} ${opt} ARM_NN_ENABLE_F32=1 ARM_NN_ENABLE_F16=${f16}"
echo "compiler: $(arm-none-eabi-gcc --version | head -n 1)"
echo "refs: ${refs[*]} -> ${labels[*]}"
echo

for i in "${!labels[@]}"; do
  echo "== function map at ${labels[$i]} (nm -S --size-sort; size in hex bytes)"
  for src in "${objects[@]}"; do
    obj="${work}/${labels[$i]}/obj/$(echo "${src}" | tr / _).o"
    echo "-- ${src}"
    arm-none-eabi-nm -S --size-sort "${obj}" | sed 's/^/   /'
  done
  echo
done

echo "== images (bytes: text data bss; delta of text+data against ${labels[0]})"
printf '%-8s' "program"
for label in "${labels[@]}"; do printf '  %-26s' "${label}"; done
printf '\n'
for p in "${programs[@]}"; do
  printf '%-8s' "${p}"
  base_flash=""
  for label in "${labels[@]}"; do
    read -r text data bss _ < <(arm-none-eabi-size -B "${work}/${label}/elf/${p}.elf" | awk 'NR == 2')
    flash=$((text + data))
    if [[ -z "${base_flash}" ]]; then
      base_flash="${flash}"
      printf '  %-26s' "${text} ${data} ${bss}"
    else
      printf '  %-26s' "${text} ${data} ${bss} ($((flash - base_flash)))"
    fi
  done
  printf '\n'
done
echo

echo "== defined functions each image gains (+) or loses (-) against ${labels[0]}"
for p in "${programs[@]}"; do
  for label in "${labels[@]:1}"; do
    diff_lines="$(comm -3 "${work}/${labels[0]}/elf/${p}.functions" "${work}/${label}/elf/${p}.functions" \
      | awk -F '\t' '{ if ($1 == "") print "+ " $2; else print "- " $1 }')"
    echo "-- ${p} at ${label}: ${diff_lines:-(none)}" | sed '2,$s/^/   /'
  done
done
