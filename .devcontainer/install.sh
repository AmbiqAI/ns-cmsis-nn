#!/bin/bash

pushd ./Tests/UnitTest
./build_and_run_tests.sh -b -r
popd

echo "Finish installing."
