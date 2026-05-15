# neuralSPOT-X

If you're already on [neuralSPOT-X (NSX)](https://github.com/AmbiqAI/neuralSPOT),
you don't need to integrate heliaCORE — NSX already does it for you.

## What NSX provides

- A CMake target (`nsx::cmsis_nn` via NSX's graph) that resolves to the
  heliaCORE-built archive matched to your NSX target arch.
- Board-flag propagation through the `NSX_BOARD_FLAGS_TARGET` interface
  library — the kernels link against the same `-mcpu`/`-mfpu` flags
  the rest of your firmware uses.
- Automatic linkage of `nsx::cmsis_nn` into your model runtime
  (heliaRT, LiteRT-for-Micro, etc.).

## Pinning a heliaCORE version inside NSX

NSX projects pin heliaCORE the same way they pin any other dependency:
via the NSX manifest or a CMake cache variable. Refer to your NSX
project's docs for the exact knob — typically:

```cmake
set(NSX_CMSIS_NN_VERSION "7.25.0" CACHE STRING "heliaCORE release to use")
```

## Verifying the link

NSX's CI includes a smoke test that links a tiny consumer against
`nsx::cmsis_nn` + the NSX board flags target, the same way real NSX
firmware does. The test config lives at
[`cmake/tests/nsx_prebuilt_real_link/`](https://github.com/AmbiqAI/ns-cmsis-nn/tree/main/cmake/tests/nsx_prebuilt_real_link)
inside this repo and runs on every heliaCORE PR.

If you ever see "undefined reference" to a CMSIS-NN symbol from an NSX
project, the most common culprits are:

- A heliaCORE version pinned older than the NSX commit you're on.
- A board-flag mismatch — NSX is targeting a different `-mcpu` than
  the heliaCORE archive was built for. `find_package` will catch
  this; see [Toolchain Pinning](../guides/toolchains.md).
