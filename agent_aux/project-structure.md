# Project structure (parallel_admm)

Overview derived from CMake and repo layout.

- **仿真执行链路**（每帧调用顺序、ADMM/XPBD、碰撞与接触）：见 [simulation-execution-chain.md](simulation-execution-chain.md)。
- **代码规整与优化清单**（拼写、重复代码、逻辑错误、main 拆分、TODO 等）：见 [code-cleanup-and-optimization.md](code-cleanup-and-optimization.md)。
- **ADMM Solver CPU/GPU 映射**（文件、依赖、设计评估与重构方向）：见 [admm-solver-cpu-gpu-mapping.md](admm-solver-cpu-gpu-mapping.md)。
- **Solver CPU/GPU 重构方案与设计**（接口、目录、分阶段实施、数据流约定）：见 [solver-cpu-gpu-refactoring-design.md](solver-cpu-gpu-refactoring-design.md)。
- **构建流程**（标准构建步骤与脚本）：见 [build-flow.md](build-flow.md)。
- **构建阻碍与修改记录**（首次构建遇到的问题及修复）：见 [build-obstacles-and-fixes.md](build-obstacles-and-fixes.md)。
- **模块化与 mediator 拆散**（constraint/contact/config/app 等归属）：见 [module-modularization-design.md](module-modularization-design.md)。

## Root

| Path | Role |
|------|------|
| `CMakeLists.txt` | Single top-level CMake; builds executable `main`. |
| `cmake/` | Custom modules: `eigen.cmake`, `libigl.cmake`, `mshio.cmake`, `tomlplusplus.cmake`, `argparse.cmake` (fetch/configure deps). |
| `cmake.in/` | `src_config.h.in` → generates `config/src_config.h` (`PROJECT_ROOT_PATH`, `PROJECT_RESOURCE_PATH`). |
| `3rdparty/polyscope` | Submodule; visualization. |
| `resource/` | Scene TOMLs, meshes, etc. (`scene/`, `test_scene/`, `paper_scene/`, `pscene/`). |

## src/

| Dir | Role |
|-----|------|
| **solver/** | ADMM full/parallel solvers, XPBD, contact sub-solver, Jacobi, sparse matrix, animator; CUDA wrappers and contact solver in `.cu`. |
| **constraint/** | Triangle, tetrahedral, bending, pin constraints; XPBD constraints; device-side code in `.cu` (e.g. `*_device.cu`). |
| **contact/** | Collision: broad/narrow phase, Embree/simple BVH, LBVH (`.cu`), contact info and utilities. |
| **mesh/** | Mesh container and physics material. |
| **mediator/** | （计划移除）Mesh→constraint, TOML→config, DCD 校验等将迁入 config/, app/, constraint/, solver/, contact/, mesh/，见 module-modularization-design.md。 |
| **mutils/** | Timer, cnpy, color, formatting, exception, QR/SVD, CUDA/GPU helpers, common types. |
| **zensim/** | Headers only: math (matrix, bit, Vec, etc.), types, meta, geometry; GLOB only picks `math/matrix/*` and `math/bit/*`. |

## Entry and targets

- **main** (default): `src/main.cpp`; args = scene TOML path, resource directory.
- Optional (commented in CMake): `triangle`, `tetrahedral`, `twobody`, `testBVH` (each has a `src/*.cpp`).

## Dependencies (from CMake)

- **vcpkg**: TBB, Embree, ZLIB.
- **cmake modules**: Eigen, libigl (with predicates), mshio, toml++, argparse.
- **Subdirectory**: polyscope.
- No ccd submodule (not used in build).
