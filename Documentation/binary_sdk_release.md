# NS-CMSIS-NN Binary SDK Package

This package is the closed-source binary SDK distribution of `ns-cmsis-nn`.

It includes:

- `include/` public headers
- `lib/` prebuilt static libraries for the supported architecture/toolchain matrix
- `cmake/` package configuration files for imported-target consumption
- `manifest.json` package and variant metadata
- `checksums.txt` package checksums

The package does not include `Source/` or implementation `.c` files.

Consumers should select the library that matches their target CPU and toolchain.
The shipped CMake package files perform the same mapping for CMake consumers and
fail if the current toolchain or target is outside the supported release matrix.
