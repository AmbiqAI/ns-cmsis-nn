#!/usr/bin/env bash

set -euo pipefail

canonicalize_path() {
  local target="$1"

  if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
    readlink -f "${target}"
    return 0
  fi

  python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${target}"
}

validate_root() {
  local root="$1"

  [[ -d "${root}" ]] || return 1
  [[ -x "${root}/bin/armclang" ]] || return 1
  [[ -d "${root}/lib" || -d "${root}/include" ]] || return 1
}

emit_root_from_binary() {
  local binary="$1"
  local real_binary
  local root

  real_binary="$(canonicalize_path "${binary}")"
  root="$(dirname "$(dirname "${real_binary}")")"

  if validate_root "${root}"; then
    printf '%s\n' "${root}"
    return 0
  fi

  return 1
}

if command -v armclang >/dev/null 2>&1; then
  emit_root_from_binary "$(command -v armclang)" && exit 0
fi

search_roots=()
[[ -n "${GITHUB_WORKSPACE:-}" ]] && search_roots+=("${GITHUB_WORKSPACE}")
[[ -n "${RUNNER_TEMP:-}" ]] && search_roots+=("${RUNNER_TEMP}")
search_roots+=("${PWD}")

for search_root in "${search_roots[@]}"; do
  [[ -d "${search_root}" ]] || continue

  while IFS= read -r candidate; do
    emit_root_from_binary "${candidate}" && exit 0
  done < <(find "${search_root}" -type f -name armclang -path '*/bin/armclang' 2>/dev/null | sort)
done

echo "Unable to locate armclang after vcpkg installation" >&2
exit 1
