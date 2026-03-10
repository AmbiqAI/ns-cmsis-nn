# Sqrt/Rsqrt s16 Unit Tests

This doc explains how to regenerate test data, build, and run the `arm_sqrt_s16` and `arm_rsqrt_s16` unit tests.

## Test Data Generation

From repo root:

```bash
python3 Tests/UnitTest/generate_sqrt_s16_test_data.py
python3 Tests/UnitTest/generate_rsqrt_s16_test_data.py
```

Generated data goes to:
- `Tests/UnitTest/TestCases/TestData/sqrt_*_s16/`
- `Tests/UnitTest/TestCases/TestData/rsqrt_float_tensor_s16/`

## Build (CMake)

If you already configured `build-cortex-m55-gcc`, just build the targets:

```bash
cd Tests/UnitTest
cmake --build build-cortex-m55-gcc --target test_arm_sqrt_s16
cmake --build build-cortex-m55-gcc --target test_arm_rsqrt_s16
```

If not configured yet:

```bash
cd Tests/UnitTest
cmake -S . -B build-cortex-m55-gcc \
  -DTARGET_CPU=cortex-m55 \
  -DCMSIS_PATH=downloads/CMSIS_5 \
  -DCMSIS_OPTIMIZATION_LEVEL=-Ofast
```

## Run (FVP)

Sqrt:

```bash
FVP_Corstone_SSE-300_Ethos-U55 -C mps3_board.uart0.shutdown_on_eot=1 \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file="-" \
  -C mps3_board.uart0.unbuffered_output=1 \
  build-cortex-m55-gcc/TestCases/test_arm_sqrt_s16/test_arm_sqrt_s16.elf
```

Rsqrt:

```bash
FVP_Corstone_SSE-300_Ethos-U55 -C mps3_board.uart0.shutdown_on_eot=1 \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file="-" \
  -C mps3_board.uart0.unbuffered_output=1 \
  build-cortex-m55-gcc/TestCases/test_arm_rsqrt_s16/test_arm_rsqrt_s16.elf
```

## Notes

- The tests run on Cortex-M55, so you must use FVP or real hardware.
- The build may print linker warnings about `_getpid` / `_kill` and RWX segments. These are expected for bare-metal builds.
