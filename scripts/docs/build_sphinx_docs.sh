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
  DOXYGEN_URL                Override Doxygen tarball URL.
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
    log "Downloading Doxygen ${doxygen_version} to ${install_root}"
    archive="${install_root%/}/doxygen-${doxygen_version}.linux.bin.tar.gz"
    url="${DOXYGEN_URL:-https://www.doxygen.nl/files/doxygen-${doxygen_version}.linux.bin.tar.gz}"

    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "${url}" -o "${archive}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${archive}" "${url}"
    else
      echo "Neither curl nor wget is available to download Doxygen." >&2
      exit 1
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
