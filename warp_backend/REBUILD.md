# Rebuilding warpsycl.dll

The bundled `native/warpsycl.dll` is built from `native/sycl_runtime.cpp`
(oneAPI DPC++ / icx). Rebuild only when changing the backend.

## Toolchain (Windows)

- MSVC Build Tools (vcvarsall.bat)
- Intel oneAPI 2025.x (icx) — the SAME major version the dll was built with;
  mixing runtimes is what breaks torch-xpu coexistence (sycl8.dll collision)

## Build

    icx /nologo /fsycl /EHsc /O2 /MD /LD /DWP_SYCL_BUILDING_RUNTIME ^
        /I<warp-sycl repo>/warp/native ^
        native/sycl_runtime.cpp ^
        /Fe:native/warpsycl.dll

## Verify after rebuild

    python -m mjlab_sycl install        # overlays + verifies device
    python tools gates: e2e + mujoco_warp agreement vs cpu (see README-SYCL-TRAINING.md)
