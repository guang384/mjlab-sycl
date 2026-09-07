# Rebuilding warpsycl.dll

The shipped `src/mjlab_sycl/backend/warpsycl.dll` is built from
`src/mjlab_sycl/backend/native/sycl_runtime.cpp` (oneAPI DPC++ / icx). The
runtime is self-contained — standard C++ plus `<sycl/sycl.hpp>` only, no warp
headers — so no extra include dirs are needed. Rebuild only when changing
the backend.

## Toolchain (Windows)

- MSVC Build Tools (vcvarsall.bat)
- Intel oneAPI 2025.x (icx) — the SAME major version the dll was built with;
  mixing runtimes is what breaks torch-xpu coexistence (sycl8.dll collision)

## Build (from the repo root)

    icx /nologo /fsycl /EHsc /O2 /MD /LD /DWP_SYCL_BUILDING_RUNTIME ^
        src/mjlab_sycl/backend/native/sycl_runtime.cpp ^
        /Fe:src/mjlab_sycl/backend/warpsycl.dll

## Verify after rebuild

    python -m mjlab_sycl install        # overlays the rebuilt backend, verifies the device
    mjlab-sycl-test                     # both gates: backend e2e, then mujoco_warp vs cpu