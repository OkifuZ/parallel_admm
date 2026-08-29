# parallel_admm

Implementation for *Efficient Frictional Contacts for Soft Body Dynamics via ADMM*, CGI 2024 (submitted to The Visual Computer).

## Solver Architecture (CPU/GPU)

The ADMM solver uses a backend abstraction (`IADMMBackend`) so the main loop works uniformly with either CPU or GPU:

- **ADMMSolver**: holds an `IADMMBackend` and delegates `init`, `step`, `reset`, etc.
- **ADMMBackendCPU**: Eigen LLT, TBB prox, ProximalQuery (host-only)
- **ADMMBackendGPU**: Jacobi iterative, CUDA prox, BVH_GPU (device-side solve, sync at step boundaries)

Scene TOML option `use_GPU = true/false` selects the backend. See `agent_aux/solver-cpu-gpu-refactoring-design.md` for details.

### Configuration flow (typed, no casts)

```
APPConfig (TOML)
  → build_solver_config()        # mediator/solver_config_applier.h — pure data
  → ADMMSolverConfig             # solver/core/admm_config.h
  → ADMMSolver::apply_config()   # facade → IADMMBackend::apply_config() → impl
```

- Unified creation: `create_solver(SolverType)` in `solver/admm_solver_factory.h`
  (`SolverType::ADMM_CPU | ADMM_GPU`; extensible for future solvers).
- After constraint assembly the app layer calls
  `ADMMSolver::set_constraint_dim()` + `ADMMSolver::finalize_constraints()`
  (GPU backend converts host constraints to device there).

### Collision detection (interface)

`ICollisionDetector` (`contact/icollision_detector.h`) is the host-facing
collision contract; `ProximalQuery` (CPU) and `BVH_GPU` (GPU) implement it.
Visualization and stats go through the interface.

### XPBD

A second, independent solver is implemented: `XPBDSolver : Solver`
(`solver/xpbd_solver.{h,cpp}`, Macklin et al. 2016 formulas). It reuses the
generic Solver state, `ICollisionDetector`, and the XPBD constraint module
(`src/constraint/xpbd/`, revived from the archive). Select it with
`[solver] type = "xpbd"` (legacy `[xpbd] enable = true` also works).
The old ADMM-parasitic XPBD implementation stays in `archive/xpbd/`;
`XPBD_utils` math helpers live in `src/constraint/xpbd_utils.h`.
See `agent_aux/xpbd-integration-plan.md`.

### Comparing solvers (headless batch)

Per-step metrics (frame, step_ms, n_contacts) are collected by every solver
and written to `<out_file>/step_metrics.csv`. Batch OBJ export is enabled
with the top-level `export_obj = true` (plus `show_windows = false` and
`end_frame = N` for headless runs).

`tools/compare_solvers.py` automates an ADMM-vs-XPBD run of one scene:

```bash
python tools/compare_solvers.py \
    --scene resource/scene/test_XPBD_ADMM.toml \
    --resource resource \
    --solvers admm,xpbd \
    --frames 30 \
    --exe build/Release/main.exe \
    --out build/compare_out
```

It patches the scene for each solver (headless + export), runs both, then
writes `frames.csv` (per-frame mean/max vertex difference, contact counts,
step times) and `compare.png` (requires matplotlib).

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
