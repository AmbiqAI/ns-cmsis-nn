# NS-CMSIS-NN Zephyr Binary Package

This package is the binary-only Zephyr distribution of `ns-cmsis-nn`.

It is not a source module. Consumers add the package to their Zephyr workspace as
an external module and Zephyr links the correct prebuilt static library for the
active target and toolchain.

The package includes:

- `zephyr/module.yml`
- `zephyr/CMakeLists.txt`
- `zephyr/Kconfig`
- `include/`
- `lib/`
- `manifest.json`

The package does not include `Source/` or any implementation `.c` files.

Selection behavior:

- GNU Arm Embedded builds map to the packaged `gcc` variants
- Armclang builds map to the packaged `armclang` variants
- ATFE builds map to the packaged `atfe` variants
- Cortex-M0, Cortex-M4 with FPU, and Cortex-M55 are supported

Unsupported toolchain or target combinations fail at configure time with a clear
error message.
