# parallel_admm

Implementation for *Efficient Frictional Contacts for Soft Body Dynamics via ADMM*, CGI 2024 (submitted to The Visual Computer).

## Solver Architecture (CPU/GPU)

The ADMM solver uses a backend abstraction (`IADMMBackend`) so the main loop works uniformly with either CPU or GPU:

- **ADMMSolver**: holds an `IADMMBackend` and delegates `init`, `step`, `reset`, etc.
- **ADMMBackendCPU**: Eigen LLT, TBB prox, ProximalQuery (host-only)
- **ADMMBackendGPU**: Jacobi iterative, CUDA prox, BVH_GPU (device-side solve, sync at step boundaries)

Scene TOML option `use_GPU = true/false` selects the backend. See `agent_aux/solver-cpu-gpu-refactoring-design.md` for details.

## Dependencies and submodules

```bash
git submodule add https://github.com/nmwsharp/polyscope.git ./3rdparty/polyscope
git submodule update --init --recursive
```

Requires [vcpkg](https://vcpkg.io/) with `VCPKG_ROOT` set. CMake pulls Eigen, libigl, TBB, Embree, ZLIB, toml++, argparse, etc. via vcpkg.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Output: `main` (or `main.exe`).

## Usage

```bash
./main <config_file_path> <resource_file_path>
```

- **config_file_path**: path to scene config (TOML), e.g. `resource/scene/bunny.toml`.
- **resource_file_path**: resource directory; mesh paths in the config are relative to this, e.g. `resource`.

Example:

```bash
./main resource/scene/bunny.toml resource
```

Scene TOML configures meshes, materials, collision, solver options, output path, etc. See examples under `resource/scene/`.
