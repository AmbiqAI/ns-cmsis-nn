#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

site_dir="site"
sphinx_src="docs"
doxygen_xml_dir="Documentation/xml"
doxygen_version="${DOXYGEN_VERSION:-1.9.6}"
install_doxygen=0
generate_doxygen=1
fast_mode=0
clean_build=1
strict=1
doctree_dir="${DOCS_DOCTREES:-build/docs/doctrees}"
sphinx_build_cmd="${SPHINXBUILD:-sphinx-build}"
doxygen_download_attempts="${DOXYGEN_DOWNLOAD_ATTEMPTS:-3}"
doxygen_retry_base_delay="${DOXYGEN_RETRY_BASE_DELAY:-2}"

usage() {
  cat <<'USAGE'
Usage: build_sphinx_docs.sh [options]

Build the heliaCORE documentation artifact:
  1. Generate Doxygen XML from the public C headers.
  2. Build Sphinx with Breathe/Exhale using the generated Doxygen XML.
  3. Verify the generated Sphinx API entrypoint exists.

Options:
  --install-doxygen          Download Doxygen 1.9.6 to /tmp if it is not on PATH.
  --doxygen-version VERSION  Doxygen version to require/install (default: 1.9.6).
  --skip-doxygen             Reuse an existing Documentation/xml/ output.
  --fast                     Fast preview build: skip Doxygen and the generated
                             C API pages, reuse caches, and drop -W. Ideal for
                             iterating on landing-page content and CSS.
  --incremental              Reuse the doctree cache and existing HTML output
                             instead of doing a clean rebuild.
  --no-strict                Do not treat Sphinx warnings as errors.
  --doctree-dir DIR          Cached doctree directory (default: build/docs/doctrees).
  --site-dir DIR             Sphinx HTML output directory (default: site).
  --sphinx-src DIR           Sphinx source directory (default: docs).
  --doxygen-xml-dir DIR      Generated Doxygen XML directory (default: Documentation/xml).
  -h, --help                 Show this help.

Environment:
  SPHINXBUILD                sphinx-build executable (default: sphinx-build).
  DOCS_DOCTREES              Cached doctree directory (default: build/docs/doctrees).
  DOXYGEN_URL                Override Doxygen tarball URL (skips the default sources).
  DOXYGEN_SHA256             Expected SHA-256 of the Doxygen tarball, for a version
                             with no pinned digest in this script.
  DOXYGEN_INSTALL_ROOT       Where to download/extract Doxygen (default: /tmp).
  DOXYGEN_DOWNLOAD_ATTEMPTS  Download attempts per source (default: 3).
  DOXYGEN_RETRY_BASE_DELAY   Seconds before the first retry; doubles each
                             attempt (default: 2).
USAGE
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Option $1 requires a value." >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-doxygen)
      install_doxygen=1
      ;;
    --doxygen-version)
      require_value "$1" "${2:-}"
      doxygen_version="$2"
      shift
      ;;
    --skip-doxygen)
      generate_doxygen=0
      ;;
    --fast)
      fast_mode=1
      generate_doxygen=0
      clean_build=0
      strict=0
      ;;
    --incremental)
      clean_build=0
      ;;
    --no-strict)
      strict=0
      ;;
    --doctree-dir)
      require_value "$1" "${2:-}"
      doctree_dir="$2"
      shift
      ;;
    --site-dir)
      require_value "$1" "${2:-}"
      site_dir="$2"
      shift
      ;;
    --sphinx-src)
      require_value "$1" "${2:-}"
      sphinx_src="$2"
      shift
      ;;
    --doxygen-xml-dir)
      require_value "$1" "${2:-}"
      doxygen_xml_dir="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  printf '\n==> %s\n' "$*"
}

# Pinned SHA-256 of the official linux-x86-64 binary tarball, keyed by version.
# The download is verified against this before anything reaches tar, so a
# truncated transfer or an error page served with HTTP 200 is rejected with a
# readable message instead of surfacing as "gzip: stdin: not in gzip format".
# Versions with no entry here fall back to a gzip magic-byte and tar-listing
# check; DOXYGEN_SHA256 pins one explicitly.
doxygen_expected_sha256() {
  case "$1" in
    1.9.6) printf '%s' '8354583f86416586d35397c8ee7e719f5aa5804140af83cf7ba39a8c5076bdb8' ;;
    *)     printf '' ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    return 1
  fi
}

# Show what the server actually sent (an error page, a redirect stub, a
# captive-portal form) rather than leaving only a downstream tar error.
dump_response_head() {
  echo "--- first 512 bytes of the response ---" >&2
  head -c 512 "$1" | tr -c '[:print:][:space:]' '.' >&2
  printf '\n--- end of response ---\n' >&2
}

# 0 if the file is a usable Doxygen tarball, 1 otherwise. Callers that are
# probing a cached file silence stderr; callers that just downloaded do not.
verify_doxygen_archive() {
  local archive="$1" expected="$2" magic actual member found

  if [[ ! -s "${archive}" ]]; then
    echo "Downloaded archive is empty: ${archive}" >&2
    return 1
  fi

  magic="$(head -c 2 "${archive}" | od -An -tx1 | tr -d ' \n')"
  if [[ "${magic}" != "1f8b" ]]; then
    echo "Payload is not gzip: expected magic 1f8b, got ${magic:-<none>} ($(wc -c <"${archive}") bytes)." >&2
    dump_response_head "${archive}"
    return 1
  fi

  if [[ -n "${expected}" ]]; then
    if ! actual="$(sha256_of "${archive}")"; then
      echo "Neither sha256sum nor shasum is available; falling back to a tar listing check." >&2
    elif [[ "${actual}" != "${expected}" ]]; then
      echo "SHA-256 mismatch for ${archive}: expected ${expected}, got ${actual}." >&2
      return 1
    else
      return 0
    fi
  fi

  # GNU tar exits 0 with an empty listing when a gzip stream holds something
  # other than a tar archive, so require the member the caller goes on to run
  # rather than treating a clean exit status as proof. grep -c reads the
  # listing to the end; grep -q would close the pipe early and, under
  # `set -o pipefail`, turn tar's SIGPIPE into a spurious verification failure.
  member="doxygen-${doxygen_version}/bin/doxygen"
  found="$(tar -tzf "${archive}" 2>/dev/null | grep -cxF -- "${member}")" || found=0
  if [[ "${found}" -eq 0 ]]; then
    echo "Archive does not contain ${member}: ${archive}" >&2
    return 1
  fi
}

fetch_url() {
  local url="$1" dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --connect-timeout 20 --max-time 900 "${url}" -o "${dest}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --timeout=20 --tries=1 -O "${dest}" "${url}"
  else
    echo "Neither curl nor wget is available to download Doxygen." >&2
    exit 1
  fi
}

# Try each source up to doxygen_download_attempts times with exponential
# backoff. The retry wraps verification as well as the transfer, because the
# failure this exists for -- a non-gzip body served with a success status --
# is one the HTTP client itself reports as a completed download.
download_doxygen_archive() {
  local archive="$1" expected="$2"
  shift 2
  local urls=("$@")
  local url attempt delay partial="${archive}.part"

  for url in "${urls[@]}"; do
    delay="${doxygen_retry_base_delay}"
    for ((attempt = 1; attempt <= doxygen_download_attempts; attempt++)); do
      log "Fetching ${url} (attempt ${attempt}/${doxygen_download_attempts})"
      rm -f "${partial}"
      if fetch_url "${url}" "${partial}"; then
        if verify_doxygen_archive "${partial}" "${expected}"; then
          mv -f "${partial}" "${archive}"
          return 0
        fi
        echo "Verification failed for the payload from ${url}." >&2
      else
        echo "Transfer failed from ${url}." >&2
      fi
      rm -f "${partial}"
      if ((attempt < doxygen_download_attempts)); then
        echo "Retrying in ${delay}s." >&2
        sleep "${delay}"
        delay=$((delay * 2))
      fi
    done
    echo "Giving up on ${url} after ${doxygen_download_attempts} attempts." >&2
  done

  return 1
}

ensure_doxygen() {
  if command -v doxygen >/dev/null 2>&1; then
    actual_version="$(doxygen --version)"
    if [[ "${actual_version}" == "${doxygen_version}"* ]]; then
      log "Using Doxygen ${actual_version} from PATH"
      return
    fi
    echo "Found Doxygen ${actual_version}, but ${doxygen_version} is required." >&2
    if [[ ${install_doxygen} -eq 0 ]]; then
      echo "Re-run with --install-doxygen or put Doxygen ${doxygen_version} on PATH." >&2
      exit 1
    fi
  elif [[ ${install_doxygen} -eq 0 ]]; then
    echo "Doxygen ${doxygen_version} is required but was not found on PATH." >&2
    echo "Re-run with --install-doxygen or install Doxygen ${doxygen_version}." >&2
    exit 1
  fi

  install_root="${DOXYGEN_INSTALL_ROOT:-/tmp}"
  doxygen_dir="${install_root%/}/doxygen-${doxygen_version}"
  doxygen_bin="${doxygen_dir}/bin/doxygen"

  if [[ ! -x "${doxygen_bin}" ]]; then
    archive="${install_root%/}/doxygen-${doxygen_version}.linux.bin.tar.gz"
    expected_sha256="${DOXYGEN_SHA256:-$(doxygen_expected_sha256 "${doxygen_version}")}"

    # The GitHub release asset is listed first: it is the upstream project's
    # own upload for the same release, served by GitHub's CDN, and
    # www.doxygen.nl/files/ prunes older releases -- it answers 404 for the
    # 1.9.6 and 1.9.8 tarballs while still serving 1.10.0 and later. Pinning an
    # older version therefore makes doxygen.nl the weaker of the two sources.
    # Whichever source answers, the payload has to clear the same verification.
    github_tag="Release_${doxygen_version//./_}"
    if [[ -n "${DOXYGEN_URL:-}" ]]; then
      download_urls=("${DOXYGEN_URL}")
    else
      download_urls=(
        "https://github.com/doxygen/doxygen/releases/download/${github_tag}/doxygen-${doxygen_version}.linux.bin.tar.gz"
        "https://www.doxygen.nl/files/doxygen-${doxygen_version}.linux.bin.tar.gz"
      )
    fi

    # An archive left by an earlier run (or restored from a CI cache) is reused
    # only if it still verifies, so a half-written file cannot become sticky.
    if [[ -f "${archive}" ]] && verify_doxygen_archive "${archive}" "${expected_sha256}" 2>/dev/null; then
      log "Reusing verified Doxygen ${doxygen_version} archive at ${archive}"
    else
      if [[ -f "${archive}" ]]; then
        log "Discarding unverifiable cached archive ${archive}"
        rm -f "${archive}"
      fi
      log "Downloading Doxygen ${doxygen_version} to ${install_root}"
      if [[ -z "${expected_sha256}" ]]; then
        echo "No pinned SHA-256 for Doxygen ${doxygen_version}; verifying the payload is a readable gzip tarball only." >&2
        echo "Set DOXYGEN_SHA256 to pin this version." >&2
      fi
      if ! download_doxygen_archive "${archive}" "${expected_sha256}" "${download_urls[@]}"; then
        echo "Failed to download a verified Doxygen ${doxygen_version} archive from any source." >&2
        exit 1
      fi
    fi

    tar -C "${install_root}" -xzf "${archive}"
  fi

  export PATH="${doxygen_dir}/bin:${PATH}"
  actual_version="$(doxygen --version)"
  if [[ "${actual_version}" != "${doxygen_version}"* ]]; then
    echo "Installed Doxygen ${actual_version}, expected ${doxygen_version}." >&2
    exit 1
  fi
  log "Using Doxygen ${actual_version} from ${doxygen_dir}"
}

if [[ ${generate_doxygen} -eq 1 ]]; then
  ensure_doxygen
  log "Generating Doxygen API XML"
  ./Documentation/Doxygen/gen_doc.sh -s
else
  log "Skipping Doxygen generation; reusing ${doxygen_xml_dir}"
fi

if [[ ${fast_mode} -eq 1 ]]; then
  export DOCS_FAST=1
  log "Fast mode: skipping the generated C API pages (DOCS_FAST=1)"
elif [[ ! -f "${doxygen_xml_dir}/index.xml" ]]; then
  echo "Doxygen XML index not found: ${doxygen_xml_dir}/index.xml" >&2
  exit 1
fi

if ! command -v "${sphinx_build_cmd}" >/dev/null 2>&1; then
  echo "Sphinx executable not found: ${sphinx_build_cmd}" >&2
  echo "Install docs dependencies with: pip install -r docs/requirements.txt" >&2
  exit 1
fi

if [[ -z "${site_dir}" || "${site_dir}" == "/" ]]; then
  echo "Refusing unsafe --site-dir value: ${site_dir}" >&2
  exit 1
fi

generated_api_dir="${sphinx_src%/}/api"
mkdir -p "${generated_api_dir}"
if [[ ${clean_build} -eq 1 ]]; then
  find "${generated_api_dir}" -mindepth 1 ! -name .gitignore -exec rm -rf {} +
  rm -rf "${site_dir}"
fi
mkdir -p "${doctree_dir}"

log "Generating Sphinx API pages (non-fatal pass)"
if ! "${sphinx_build_cmd}" -b html --keep-going "${sphinx_src}" "${site_dir}"; then
  log "Non-fatal generation pass reported warnings/errors; continuing to post-process generated API pages"
fi

log "Normalizing generated doxygenfunction directives for softmax helpers"
python - <<'PY'
from pathlib import Path

api_dir = Path("docs/api")
if not api_dir.exists():
  raise SystemExit(0)

replacements = {
  ".. doxygenfunction:: arm_nn_softmax_1x2_f16(const float16_t, float16_t)":
    ".. doxygenfunction:: arm_nn_softmax_1x2_f16",
  ".. doxygenfunction:: arm_nn_softmax_1x2_f32(const float32_t, float32_t)":
    ".. doxygenfunction:: arm_nn_softmax_1x2_f32",
}

for path in api_dir.glob("*.rst"):
  text = path.read_text(encoding="utf-8")
  updated = text
  for old, new in replacements.items():
    updated = updated.replace(old, new)
  if updated != text:
    path.write_text(updated, encoding="utf-8")
PY

rm -rf "${site_dir}"

log "Building Sphinx site (strict pass)"
"${sphinx_build_cmd}" -b html -W --keep-going "${sphinx_src}" "${site_dir}"
sphinx_args=(-b html -d "${doctree_dir}")
if [[ ${strict} -eq 1 ]]; then
  sphinx_args+=(-W --keep-going)
fi

if [[ ${fast_mode} -eq 1 ]]; then
  log "Building Sphinx site (fast preview)"
else
  log "Building Sphinx site"
fi
"${sphinx_build_cmd}" "${sphinx_args[@]}" "${sphinx_src}" "${site_dir}"

if [[ ${fast_mode} -eq 0 ]]; then
  api_index="${site_dir%/}/api/library_root.html"
  if [[ ! -f "${api_index}" ]]; then
    echo "Generated Sphinx API index not found: ${api_index}" >&2
    exit 1
  fi
  printf 'Generated API: %s\n' "${api_index}"
fi

log "Sphinx docs built successfully"
printf 'Sphinx site: %s\n' "${site_dir}"
