#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

manifest="${1:?manifest path required}"
destination="${2:?destination required}"
environment_json="${3:?environment output path required}"

for command in curl jq sha256sum tar unzip; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 2
  }
done

jq -e '
  .schema_version == 1
  and .platform == "linux-x86_64"
  and (.tools | length > 0)
  and all(.tools[];
    (.id | test("^[a-z0-9][a-z0-9-]*$"))
    and (.version | length > 0)
    and (.url | startswith("https://"))
    and (.sha256 | test("^[0-9a-f]{64}$"))
    and (.archive == "tar.gz" or .archive == "tar.xz" or .archive == "zip")
    and (.strip_components | type == "number")
    and (.probe | length > 0)
    and (.path | length > 0)
    and (.license | length > 0))
' "${manifest}" >/dev/null

rm -rf "${destination}"
mkdir -p "${destination}"
download_dir="$(mktemp -d)"
trap 'rm -rf "${download_dir}"' EXIT

while IFS=$'\t' read -r id version url expected archive strip_components probe; do
  install_dir="${destination}/${id}"
  download="${download_dir}/${id}.${archive}"
  mkdir -p "${install_dir}"
  curl --fail --location --silent --show-error --retry 3 --retry-all-errors \
    "${url}" --output "${download}"
  printf '%s  %s\n' "${expected}" "${download}" | sha256sum --check -

  case "${archive}" in
    tar.gz)
      if tar --list --gzip --file "${download}" \
        | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
        echo "${id}: archive contains an unsafe path" >&2
        exit 2
      fi
      tar --extract --gzip --file "${download}" --directory "${install_dir}" \
        --strip-components="${strip_components}" --no-same-owner
      ;;
    tar.xz)
      if tar --list --xz --file "${download}" \
        | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
        echo "${id}: archive contains an unsafe path" >&2
        exit 2
      fi
      tar --extract --xz --file "${download}" --directory "${install_dir}" \
        --strip-components="${strip_components}" --no-same-owner
      ;;
    zip)
      [[ "${strip_components}" -eq 0 ]] || {
        echo "${id}: zip archives do not support strip_components" >&2
        exit 2
      }
      if unzip -Z1 "${download}" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
        echo "${id}: archive contains an unsafe path" >&2
        exit 2
      fi
      unzip -q "${download}" -d "${install_dir}"
      ;;
  esac

  [[ -x "${install_dir}/${probe}" ]] || {
    echo "${id} ${version}: missing executable probe ${probe}" >&2
    exit 3
  }
done < <(
  jq -r '.tools[] | [
    .id, .version, .url, .sha256, .archive,
    (.strip_components | tostring), .probe
  ] | @tsv' "${manifest}"
)

jq --arg root "${destination}" '
  {
    schema_version: 1,
    platform: .platform,
    paths: {
      PATH: [.tools[] | "\($root)/\(.id)/\(.path)"]
    },
    tools: (
      reduce .tools[] as $tool ({};
        reduce (($tool.environment // {}) | to_entries[]) as $entry (.;
          .[$entry.key] = "\($root)/\($tool.id)/\($entry.value)"))
    ),
    provenance: [
      .tools[] | {
        id, version, url, sha256, license,
        runtime_license: (.runtime_license // null)
      }
    ]
  }
' "${manifest}" > "${environment_json}"

jq -e '
  .schema_version == 1
  and (.paths.PATH | length > 0)
  and (.provenance | length > 0)
' "${environment_json}" >/dev/null
