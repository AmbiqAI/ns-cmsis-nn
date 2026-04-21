# NS-CMSIS-NN Zephyr Module Package

This package is the Zephyr distribution of `ns-cmsis-nn`.

It is a source-module package, not a prebuilt binary SDK. Consumers should add the
package to their Zephyr workspace as a module and let Zephyr compile the sources as
part of the application build.

The package includes:

- `zephyr/module.yml`
- `zephyr/CMakeLists.txt`
- `zephyr/Kconfig`
- `Source/`
- `Include/`

The Zephyr module expects the normal Zephyr CMSIS integration to be present and uses
Zephyr's module discovery/build flow rather than linking against a separate
Zephyr-specific static library.
