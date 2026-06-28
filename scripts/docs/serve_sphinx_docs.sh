#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

host="${DOCS_HOST:-127.0.0.1}"
port="${DOCS_PORT:-8014}"
venv_dir="${DOCS_VENV:-/tmp/sphinx-docs-venv}"
site_dir="site"
doctree_dir="${DOCS_DOCTREES:-build/docs/doctrees}"
build_doxygen_arg="--install-doxygen"
install_deps=1
fast_mode=0
watch_mode=0

usage() {
  cat <<'USAGE'
Usage: serve_sphinx_docs.sh [options]

Build and serve the heliaCORE Sphinx documentation locally.

Default one-line preview:
  bash scripts/docs/serve_sphinx_docs.sh

Fastest iteration on landing-page content and CSS (live reload, no Doxygen):
  bash scripts/docs/serve_sphinx_docs.sh --watch --fast

Options:
  --host HOST          Bind host for the preview server (default: 127.0.0.1).
  --port PORT          Bind port for the preview server (default: 8014).
  --venv DIR           Python virtual environment directory (default: /tmp/sphinx-docs-venv).
  --site-dir DIR       Sphinx HTML output directory (default: site).
  --watch              Live-reload server via sphinx-autobuild (incremental rebuilds).
  --fast               Skip Doxygen and the generated C API pages for quick previews.
  --skip-doxygen       Reuse existing Documentation/xml/ for a faster rebuild.
  --no-install-deps    Reuse the virtual environment without installing docs requirements.
  -h, --help           Show this help.

Environment:
  DOCS_HOST            Same as --host.
  DOCS_PORT            Same as --port.
  DOCS_VENV            Same as --venv.
  DOCS_DOCTREES        Cached doctree directory (default: build/docs/doctrees).
  DOXYGEN_URL          Override Doxygen tarball URL used by build_sphinx_docs.sh.
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
    --host)
      require_value "$1" "${2:-}"
      host="$2"
      shift
      ;;
    --port)
      require_value "$1" "${2:-}"
      port="$2"
      shift
      ;;
    --venv)
      require_value "$1" "${2:-}"
      venv_dir="$2"
      shift
      ;;
    --site-dir)
      require_value "$1" "${2:-}"
      site_dir="$2"
      shift
      ;;
    --skip-doxygen)
      build_doxygen_arg="--skip-doxygen"
      ;;
    --fast)
      fast_mode=1
      ;;
    --watch)
      watch_mode=1
      ;;
    --no-install-deps)
      install_deps=0
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to build and serve the docs." >&2
  exit 1
fi

if [[ -z "${site_dir}" || "${site_dir}" == "/" ]]; then
  echo "Refusing unsafe --site-dir value: ${site_dir}" >&2
  exit 1
fi

if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

if [[ ${install_deps} -eq 1 ]]; then
  "${venv_dir}/bin/python" -m pip install --upgrade pip
  "${venv_dir}/bin/pip" install -r docs/requirements.txt
fi

build_args=()
if [[ ${fast_mode} -eq 1 ]]; then
  build_args+=(--fast)
else
  build_args+=("${build_doxygen_arg}")
fi

if [[ ${watch_mode} -eq 1 ]]; then
  # Live-reload iteration loop: build once, then rebuild incrementally on change.
  # The generated C API tree and HTML output are ignored to avoid rebuild loops.
  SPHINXBUILD="${venv_dir}/bin/sphinx-build" \
    bash scripts/docs/build_sphinx_docs.sh "${build_args[@]}" --incremental \
      --doctree-dir "${doctree_dir}" --site-dir "${site_dir}"

  autobuild_env=()
  sphinx_args=(-b html -d "${doctree_dir}")
  if [[ ${fast_mode} -eq 1 ]]; then
    autobuild_env=(env DOCS_FAST=1)
  else
    sphinx_args+=(-W --keep-going)
  fi

  url="http://${host}:${port}/"
  printf '\nWatching heliaCORE docs (live reload) at %s\n\n' "${url}"
  exec "${autobuild_env[@]}" "${venv_dir}/bin/sphinx-autobuild" \
    "${sphinx_args[@]}" \
    --host "${host}" --port "${port}" \
    --ignore "*/api/*" \
    --ignore "*/_build/*" \
    --re-ignore '.*\.doctree' \
    --watch docs/_static \
    --watch docs/_ext \
    docs "${site_dir}"
fi

SPHINXBUILD="${venv_dir}/bin/sphinx-build" \
  bash scripts/docs/build_sphinx_docs.sh "${build_args[@]}" \
    --doctree-dir "${doctree_dir}" --site-dir "${site_dir}"

url="http://${host}:${port}/"
printf '\nServing heliaCORE docs at %s\n' "${url}"
if [[ ${fast_mode} -eq 0 ]]; then
  printf 'Generated API: %sapi/library_root.html\n' "${url}"
fi
printf '\n'
exec python3 -m http.server "${port}" --bind "${host}" --directory "${site_dir}"
