# ADMM Solver：CPU / GPU 文件与依赖映射

供后续重构参考。

---

## 1. 分支逻辑

**main.cpp**、**toml_to_config.h**：
- `config.use_GPU == true` → `ADMMParallelSolver`（GPU）
- `config.use_GPU == false` → `ADMMSolverFull_RL_damping`（CPU）

---

## 2. CPU Solver

| 角色 | 文件 | 说明 |
|------|------|------|
| 基类 | `solver/admm_full_solver.h` | `ADMMSolverFull`，定义通用接口与数据结构 |
| CPU 实现 | `solver/admm_full_solver_RL_damping.cpp` | `ADMMSolverFull_RL_damping`：`init`, `precompute`, `step`, `step_fast`, `project_feasible`, `compute_Scc` |
| 共用 | `solver/admm_full_solver.cpp` | `ADMMSolverFull::init`, `reset`, `project_feasible`（部分）、`sequential_greedy_coloring` |

**CPU 自实现依赖：**
- Eigen（稀疏矩阵、SimplicialLLT、Triplet）
- TBB（parallel_for）
- `ProximalQuery`（CPU 碰撞）
- `contact/narrow_phase`（CPU 接触检测）

---

## 3. GPU Solver

| 角色 | 文件 | 说明 |
|------|------|------|
| 类声明 | `solver/parallel_solver.h` | `ADMMParallelSolver` 继承 `ADMMSolverFull_RL_damping`，`ContactDataDevice` |
| 实现 | `solver/admm_parallel_global_solver.cu` | 重写 `init`, `precompute`, `step`，以及 `convert_constraint2device` |

**GPU 自实现依赖：**

| 文件 | 职责 |
|------|------|
| `solver/jacobi_solver.h` + `jacobi_solver.cu` | `CuCompactSparseMat`, `CuSolverData`, `cu_jacobi_global`, `copy_mat2thrustvector` 等 |
| `solver/compact_sparse_matrix_util.cu` | `op_Ax`, `op_a_plus_b`, `op_a_minus_b`, `op_scale`, `resizeThrust` |
| `solver/contact_solver.cu` | `compute_Scc_impl_cu`, `convert_DCD_info`, `project_impl` |
| `solver/mcuda_wrapper.cu` | `do_pre_integration`, `do_post_process` |
| `constraint/triangle_constraint_device.cu` | 三角约束 GPU 核 |
| `constraint/pin_constraint_device.cu` | 固定约束 GPU 核 |
| `constraint/bending_constraint_device.cuh` | 弯曲约束 GPU 核 |
| `constraint/tetrahedral_constraint_device.cuh` | 四面体约束 GPU 核 |

**外部依赖：** Thrust、CUDA、`BVH_GPU`、`contact/collision_detection_cuda`。

---

## 4. 继承关系

```
Solver (solver.h)
  └── ADMMSolverFull (admm_full_solver.h)
        └── ADMMSolverFull_RL_damping (admm_full_solver.h, admm_full_solver_RL_damping.cpp)
              └── ADMMParallelSolver (parallel_solver.h, admm_parallel_global_solver.cu)
```

---

## 5. 设计评估与改进方向

### 5.1 当前设计问题

**（1）继承耦合过重**

- `ADMMParallelSolver` 继承 `ADMMSolverFull_RL_damping`，直接拿到 CPU 路径的全部成员。
- 同时使用 `Gamma_c`, `K_c`, `involved_cid` 等 host 结构，以及 `ContactDataDevice`, `CuSolverData` 等 device 结构。
- 哪些逻辑在 host、哪些在 device 不清晰，GPU 路径夹杂大量 CPU 数据结构。

**（2）逻辑重复**

- `project_feasible`：CPU 用 `_project_feasible_plain`（TBB），GPU 用 `_project_feasible_impl` → `project_impl`（CUDA），算法相似、实现分散。
- `compute_Scc`：CPU 在 `admm_full_solver_RL_damping`，GPU 在 `compute_Scc_impl` → `compute_Scc_impl_cu`，两套实现。
- 全局求解：CPU 用 Eigen LLT 或 GS，GPU 用 Jacobi，无统一抽象。

**（3）职责边界不清**

- `precompute` 在 CPU 和 GPU 里既做稀疏矩阵组装，又做 backend 特有初始化，混在一起。
- `parallel_solver.h` 中既有 solver 逻辑，又有 `ContactDataDevice` 等底层设备结构。

**（4）依赖分散**

- 线性代数、约束投影、接触处理分散在多个 `.cu` / `.cuh`。
- 没有清晰的 “GPU 后端” 模块，调用关系靠头文件串联。

**（5）Host/Device 数据流复杂**

- `copy_mat2thrustvector`、`copy_thrustvector2mat`  scattered 在 `jacobi_solver`。
- 何时 sync、谁负责 copy 缺乏统一约定。

### 5.2 改进方向建议

| 方向 | 说明 |
|------|------|
| **策略模式 / 组合** | 用 `ADMMBackend`（CPU / GPU 实现）替代深层继承，solver 持有一个 backend，统一 `step()` 接口。 |
| **提取公共算法层** | `ADMMLoop` 只负责迭代流程，把线性求解、局部投影、接触处理抽象成接口（如 `ILinearSolver`, `ILocalProjector`, `IContactHandler`），CPU/GPU 各自实现。 |
| **分离数据结构** | CPU 只维护 host 数据，GPU 只维护 device 数据，在边界做显式 copy；避免同一 solver 同时持有两套等价结构。 |
| **统一接触接口** | `project_feasible` 和 `compute_Scc` 收敛为同一抽象（如 `IContactSolver`），CPU/GPU 各一个实现。 |
| **明确模块边界** | 例如：`solver/core/`（算法流程）、`solver/backend_cpu/`、`solver/backend_gpu/`（含 jacobi、compact_sparse、contact、constraint device）。 |
| **简化继承链** | `ADMMParallelSolver` 不必继承 `ADMMSolverFull_RL_damping`，可与 CPU solver 共享 `ADMMSolverFull` 或更高层抽象，差异由 backend 承担。 |

### 5.3 重构优先级建议

1. **高**：统一 `project_feasible` / `compute_Scc` 的接口，将 CPU/GPU 实现收敛到同一抽象。
2. **高**：把 GPU 相关实现归到 `backend_gpu/` 或类似目录，建立清晰边界。
3. **中**：用 backend 抽象替代 GPU solver 对 CPU RL_damping 的继承。
4. **中**：集中管理 Host ↔ Device copy，避免 scattered copy 逻辑。
5. **低**：调整目录与命名，使 CPU/GPU 对应关系更直观。
