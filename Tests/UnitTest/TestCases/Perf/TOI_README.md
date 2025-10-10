# 0. Devcontainer and Docker

## 0.1 Dockerfile updates
- The Dockerfile uses llvm 12 and clang-format-12 for compatibility. It used to use version 16 for these, and some other dependency issue arises. However, this simple change resolved it, and it's pretty minor.

## 0.2 Custom tflite_micro setup for non x86_64
- Right now, Pypi only contains x86_64 wheels for tflite_micro. 
  - The condition platform_machine == "x86_64" was added to requirements.txt to guard against this installation attempt and error on non x86_64 platforms.
- Currently, bazel dependency (for setting up tflite_micro on non x86_64) is in devcontainer.json, but not Dockerfile. Should add it to the Dockerfile as well. Currently not a big issue since github uses x86_64 and thus does not follow the alternate tflite_micro installation pathway.
- I mentioned that the tflite_micro wheel file has a name that changes depending on the date of the last stable commit of the tflite_micro source git repo (as it bazel-builds only off of that latest commit).
  - For example, tflite_micro-0.dev20230920161638-py3-none-any.whl, if the last stable commit was on September 20, 2023.
- I found that every time tflite_micro is built, this wheel file's name is always stored in the auto generated file {BUILD_DIR}/bazel-bin/python/tflite_micro/whl.name.
- Therefore, I was able to just cat whl.name to get the correct name regardless of what time it was built (and what the wheel is named).
- However, the way tflite_micro's README says to get the name is by using the non-built-in 'tree' command. This command is typically installed via apt (sudo-apt get tree or similar), but I just had general trouble using apt install with the devcontainer, and I found the whl.name workaround before I thought about revisitng it.
- But, in case for some unlikely reason the tflite_micro maintainers get rid of this generated whl.name file or significantly alter the build process, refer to https://github.com/tensorflow/tflite-micro/blob/main/python/tflite_micro/README.md for the build directions if in doubt.

# 1. Performance Profiling .h File

- Named cmsis_perf_profile.h, stored in ./Include/ along with arm_nn headers

## 1.1 DWT and its other metrics
- In the future, we may want to track more than merely CPU CYC count.
  - The only recorded DWT metrics so far are CYC counts and CPI.
  - There should be no need to add additional code that modifies the DWT hardware registers itself; all 6 of those are already zeroed out in perf_dwt_init. It is just that the struct "perf_ctx_t" to store perf snapshots only has those 2 fields.
  - To accomplish this, one must first modify the struct "perf_ctx_t" that holds the counters to support the others
  - Likewise, the perf_dwt_init method must initialize all the members of this struct to 0, and perf_dwt_capture and perf_dwt_delta must capture all members, not just the current 2.
  - The print statement currently on line 683 in perf_print_all must print all counters of interest
  
## 1.2 Extended Print functionality
- As mentioned, perf_print_all only supports taking in two arrays of dimensions; these are assumed to be the LHS and RHS inputs.
- For ops such as strided slice, this is insufficient to convey all information in accurate detail; so, a function signature taking in variable numbers of arguments or even additional strings that specify what those arguments are (for the name of the json field) may be better.
- For find and replace purposes, all Perf tests, without exception, either have the line(s)
  - Single input:
    - const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    - perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
  - Multiple input:
    - const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    - const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);
    - perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
  - perf_input_tensor_dims and perf_rhs_input_tensor_dims are defined earlier in the same scope as these lines on a per-test basis.
  - The date is printed out as a string field as part of the JSON, but it is currently not used functionally.

## 1.3 Including This header file
- Down the line, functionality to NOT include this header file when -P is invoked or if perf isn't needed when running all tests may be desired.

## 1.4 Cortex-M0 Series and cores without DWT
- For compatibility reasons, it may be good to surround all of the macros with an #ifdef DWT #else no-op, as currently it will cause compile errors if attempted on Cortex-M0. However, nothing would be left to profile anyway.

# 2. RefactoredTestGen and Data generation

## 2.1 Existing Issues
- Most of these are now being uncovered or already known
- Numerous convolution and LSTM data requires tflite_micro. For convolution, a cast from the numpy type 64 bit float into the normal python float type was needed. Otherwise, np.float64 would be emitted into the model json, leading to the parse error
  - This is because some of the conv data uses "json" instead of "keras" in its json field "tflite_generator". This results in another json file, this time for the model parameters, being created in the data folder for use by later steps of data generation. It is in this second json where np.float64 will appear if not downcast.
  - Even so, trying to generate data for any convolution test case that was not tiny caused a buffer overflow message.
- op_quantize and op_dequantize.py now no longer use the hardcoded 1x8x1, and instead operate based off the dimensions in the json test plan.
- I created op_softmax following the non-refactored's softmax generator and Adam's attempted relu generators. It works fine for perf tests, but its correctness is still unverified.
  - Also, int8xint16 softmax is not yet supported.
- I believe that the elementwise_mul perf test (and generator) for int16 are currently doing int16 with int8 weights (int16xint8), not completely int16. If this is not intended, it should be changed
- Some ops, such as fully connected, are just missing int16 support, but this could stem from lack of underlying python support.

# 3. Perf C tests and run_perf_test_json

## 3.1 C test and Unity Tester Structure

- All perf test C files have functions with the following naming convention (if there is no dimensionwise isolation, then that is not part of the name): void perf_arm_OPNAME_DIMENSION_DATATYPE_INPUTSIZE(void) { }

- All functions that have the same name except INPUTSIZE are grouped inside a wrapper named perf_arm_OPNAME_DIMENSION_DATATYPE() as follows:

    perf_arm_OPNAME_DIMENSION_DATATYPE () {

        perf_arm_OPNAME_DIMENSION_DATATYPE_TINY(); 

        printf(",\n");

        perf_arm_OPNAME_DIMENSION_DATATYPE_AVG(); 

        printf(",\n");

        ...
        etc.

    }

- The unity C file then has test functions called test_perf_arm_OPNAME_DIMENSION_DATATYPE() with a body of {perf_arm_OPNAME_DIMENSION_DATATYPE(); }

## 3.2 Tests that cause FVP to hang

- Look for commented out lines inside perf_arm_OPNAME_DIMENSION_DATATYPE():
  - All blocks of tests that hang are preceded by "// needs debugging" within this function
  - Inside this function, you will also see, for example:
    // perf_arm_OPNAME_DIMENSION_DATATYPE_LARGE();
    // printf(",\n");
  - It will always be lines with // followed by a space followed by "perf_arm_" and then "printf", so something such as a sed/perl command that can find and replace "// perf" and "// printf" with "perf" and "printf" can uncomment all tests once the issue is fixed.

# 4. Regression Testing

## 4.1 run_perf_test_json.sh
- A potential feature to optimize is that this script will build by calling build_and_run_tests.sh -r
  - This causes all tests, not just perf, to be built, which is more than necessary.
  - Ties into the fact that there isn't support for running only Perf and no correctness (there is for the other way around).
- On line 89, one will see the command issuing the call to parse_and_comp_perf_results.py
  - If one wants to change the default acceptable error thresholds, pass the following additional command line options to the python script call:
    - --cyc_delt=10 If, for example, a maximum of 10% increase in Cycle Count is acceptable.
    - --mac_delt=5 If, for example, a maximum of 5% deviation from expected MACs is acceptable.
- Like build_and_run_tests.sh, cortex-m55 is the default if no core is specified.
- Filename conventions:
  - The golden/reference performance file will be called perf_results_{CPU_NAME}_ref.json
    - If this file is not found to exist during running, the script will copy the results from this current run (about to be executed) to perf_results_{CPU_NAME}_ref.json and save/push that ref file for future comparisons.
    - If this file must be manually overridden or adjusted for any reason, simply modifying it and pushing it will do so. The GitHub CI should then use it in its runs thereafter.
  - The generated json file for errors will be called perf_results_{CPU_NAME}_err.json
    - It contains a list of json for ops that regressed with the format {"failed_tc":[OP_NAME_STRING], "current_perf":{JSON with current performance}, "golden_perf":{JSON with golden performance}}
    - As mentioned above, the "date" field is included in all performance JSON containing the date which produced the run; however, this currently does not have any function yet
  - The temporary file that each run inputs into (for comparison against the ref) is just called perf_results_{CPU_NAME}.json

## 4.2 parse_and_comp_perf_results.py
- By default, cyc_delt is 5 and mac_delt is 1 (so 5% and 1% tolerable thresholds.)
- By default, if no arguments are specified for the perf json filename/reference filename, m55's will be used.
- Possible to add future extended support for Per-MAC thresholds, as currently, all MACs use --mac_delt
- Another potential feature is in parse_and_comp_perf_results.py, one may want to add an argparse option to toggle on/off automatic reference/golden perf updating (for Cycle Counts). Currently, it is hardwired to on.

# 5. GitHub CI

## 5.1 Auto-Update Regression Tests
- This is currently in ci_perf.yml
  - Currently, the reference perf json is based off of GitHub's platform (since the initial run did not have a ref before, the first run on the GitHub running platform was made the standard.)
- "Concurrency" issue
  - It seems that the CI tests for cortex-m4 and m55 are parallel. This can raise rebase/merge conflict issues: if the m4 run modifies the m4 ref.json BEFORE the m55 run finishes but AFTER the m55 run has done the git checkout action, the m55 run has to stash and rebase before it pushes its updated m55 ref.json
    - It actually may not cause a merge conflict, since they are modifying spearate files (and the rebase will succeed without needing manual merge conflict intervention). However, this behavior may still warrant some cleaning up or forcing the CI tests to run on one core at a time in series and not in parallel.

# 6. Miscellaneous
- ARM.CMSIS-NN.pdsc may be out of date in the perf branch.
