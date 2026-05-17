#!/bin/bash
set -euo pipefail

python -m pip install --no-cache-dir pre-commit==3.8.0
pre-commit install

pushd ./Tests/UnitTest
./build_and_run_tests.sh -b -r
popd

echo "Finish installing."
