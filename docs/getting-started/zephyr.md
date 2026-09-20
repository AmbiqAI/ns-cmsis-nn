# Zephyr Module

heliaCORE is a first-class Zephyr module. Add it to your `west.yml`,
flip a Kconfig, and link.

## 1. Add to `west.yml`

```yaml
manifest:
  remotes:
    - name: ambiqai
      url-base: https://github.com/AmbiqAI
  projects:
    - name: ns-cmsis-nn
      remote: ambiqai
      revision: v7.35.1 # x-release-please-version
      path: modules/lib/ns-cmsis-nn
```

Then `west update`.

## 2. Enable in your app's `prj.conf`

```kconfig
CONFIG_NS_CMSIS_NN=y
CONFIG_NS_CMSIS_NN_ALL=y
```

That's the source build — Zephyr will compile the kernels with your
project's toolchain. The CMSIS-NN headers are added to the include
path automatically, and the kernels are linked into your image.

## 3. (Optional) Use the prebuilt `.a`

If you want to skip the kernel rebuild and use the GCC-built archive
from the GitHub Release:

```kconfig
CONFIG_NS_CMSIS_NN=y
CONFIG_NS_CMSIS_NN_USE_PREBUILT=y
CONFIG_NS_CMSIS_NN_PREBUILT_PATH="/path/to/extracted/ns-cmsis-nn-<cpu>-<toolchain>-<version>"
```

`CONFIG_NS_CMSIS_NN_PREBUILT_PATH` should point at the **extracted**
tarball directory (the one containing `lib/libns-cmsis-nn.a` and
`include/`). The module wires it in as an `IMPORTED STATIC` library
and applies the include directories via `zephyr_include_directories`.

:::{tip} Toolchain alignment
Your Zephyr toolchain must produce ABI-compatible objects with the archive's
CPU/FPU flags, float ABI, and calling convention. Prefer the release tarball
whose toolchain stamp matches your Zephyr compiler.
:::

## 4. Verify the module wiring

After `west build`, confirm Zephyr picked up the module and selected the path
you expected:

```bash
grep -E '^CONFIG_NS_CMSIS_NN' build/zephyr/.config
```

For a source build, the generated build graph should compile heliaCORE sources
from `modules/lib/ns-cmsis-nn/Source/`. For a prebuilt build, the final link
command should include the extracted `lib/libns-cmsis-nn.a` from
`CONFIG_NS_CMSIS_NN_PREBUILT_PATH`.

If the module does not appear in `.config`, check that the `west.yml` path
matches the module location and that your application enables
`CONFIG_NS_CMSIS_NN=y`.

## Source-build configuration ownership

Zephyr's Kconfig symbols remain authoritative, including the float-width
settings. The adapter updates both the CMake cache and normal variables from
Kconfig, so a standalone `-DARM_NN_ENABLE_F16=ON` cannot bypass a Kconfig
dependency. Optimization comes from Zephyr's `optimization_fast` setting;
standalone and NSX optimization options do not override it. Float and inline
requantize definitions remain visible to application translation units through
Zephyr's global definition API.

The shared source mapping preserves Zephyr's group order and empty selection:
when no group is enabled, no sources are attached. This is not the standalone or
NSX convention where an empty group list selects all groups. Prebuilt mode keeps
its existing manifest checks and does not apply source-build optimization.

For an application using the repository's armclang toolchain, set CMP0123 to NEW
after `cmake_minimum_required()` and before enabling C/C++ with `project()`.
The module is included too late to change compiler initialization. See the
[NSX application setup and manual check](neuralspot-x.md#arm-compiler-application-setup)
for the policy example; that modeled fixture does not establish a Zephyr board
build. Use the policy setup required by your actual Zephyr/toolchain integration.

## Reference

- Module manifest: [`zephyr/module.yml`](https://github.com/AmbiqAI/ns-cmsis-nn/blob/main/zephyr/module.yml)
- Kconfig: [`zephyr/Kconfig`](https://github.com/AmbiqAI/ns-cmsis-nn/blob/main/zephyr/Kconfig)
- CMake glue: [`zephyr/CMakeLists.txt`](https://github.com/AmbiqAI/ns-cmsis-nn/blob/main/zephyr/CMakeLists.txt)
