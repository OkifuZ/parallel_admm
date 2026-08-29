# XPBD 归档记录

**日期**: 2026-03-08（重构 Phase B 的一部分）

## 背景

原工程支持 XPBD 模式（`[xpbd] enable=true` 时 main 用 `XPBDSolver` 替代 ADMM step）。
`XPBDSolver` 通过 3 处 `reinterpret_cast<ADMMImplCPU*>(solver)` 寄生在 ADMM 实现上，
直接读写 ADMM 的 15+ 个成员——这是"其他 solver 兼容"的最大障碍。

## 决定

**XPBD 归档**：不进入生效代码，代码保留在 `archive/xpbd/`，不参与编译；
架构上保留未来接入的可能性（Solver 基类 + 碰撞接口，见 Phase A）。

## 文件去向

| 原路径 | 现路径 | 说明 |
|--------|--------|------|
| `src/solver/XPBD_solver.h` | `archive/xpbd/XPBD_solver.h` | XPBDSolver 本体 |
| `src/constraint/XPBD_constraints.h` | `archive/xpbd/XPBD_constraints.h` | XPBD 约束类 + utils（完整原文件） |
| `src/constraint/XPBD_bending.cpp` | `archive/xpbd/XPBD_bending.cpp` | 弯曲约束实现 |
| `src/constraint/XPBD_traingle.cpp` | `archive/xpbd/XPBD_traingle.cpp` | 三角形约束实现 |
| `src/constraint/XPBD_tetrahedron.cpp` | `archive/xpbd/XPBD_tetrahedron.cpp` | 四面体约束实现 |
| `src/constraint/xpbd_utils.h` | **保留在 src** | `XPBD_utils` 命名空间（仅工具函数），ADMM 的 `triangle_constraint.h` 使用 `get_FEMTriangleGradient()` |

## 生效代码清理

- `solver.h`：移除 `XPBDConstraintList` / `add_XPBDConstraints` / `m_XPBDconstraints` 及 include
- `constraint.h`：移除 `#include "XPBD_constraints.h"`
- `mesh_to_constraint.h`：`geometry_to_constraints()` 去掉 XPBD 列表参数；删除全部 XPBD 注释代码块
- `app_initializer.cpp` / `app_context.h` / `main.cpp`：移除 `enable_XPBD` / `XPBD_solver` 分支
- `collision_detection_cuda.cpp`：移除无用的 `XPBD_solver.h` / `PBD_solver.h` include
- `IContactSolver::compute_Scc()`：去掉 `is_XPBD` 参数及 CPU 实现中的 XPBD 分支
- `admm_impl_cpu.h`：移除未使用的 `m_maxXPBDIterations`
- `toml_to_config.h`：`[xpbd]` 字段保留解析（旧场景文件兼容），标记 DEPRECATED，行为上被忽略

## 恢复方法（未来）

1. 把 `archive/xpbd/` 下文件拷回 `src/` 对应位置（`XPBD_constraints.h` 放回
   `src/constraint/`，`xpbd_utils.h` 与归档版保持同一份 utils）。
2. 恢复 `solver.h` 的 XPBD 成员、`mesh_to_constraint.h` 参数、`app_initializer` 分支。
3. 建议改为独立 `XPBDSolver : Solver` 实现（经 `ICollisionDetector` 做碰撞），
   不再 `reinterpret_cast` 到 ADMMImplCPU。
