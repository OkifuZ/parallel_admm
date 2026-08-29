# Solver 相关类 CPU/GPU 重构方案与设计

基于 [admm-solver-cpu-gpu-mapping.md](admm-solver-cpu-gpu-mapping.md)、[project-structure.md](project-structure.md)、[simulation-execution-chain.md](simulation-execution-chain.md) 整理。

---

## 1. 现状与问题概览

### 1.1 当前继承链

```
Solver (solver.h)
  └── ADMMSolverFull (admm_full_solver.h)       ← Host 数据结构：SpMatf, Gamma_c, K_c 等
        └── ADMMSolverFull_RL_damping           ← CPU step_fast, compute_Scc, project_feasible
              └── ADMMParallelSolver            ← GPU step 重写，但继承全部 CPU 成员
```

### 1.2 核心问题

| 问题 | 具体表现 |
|------|----------|
| **继承耦合** | GPU solver 继承 CPU solver，同时持有 Host/Device 两套等价数据（Gamma_c/K_c vs d_Gamma_c/d_K_c） |
| **职责混杂** | `parallel_solver.h` 既有 ContactDataDevice，又有 Gamma_i、contact_islands 等 host 结构 |
| **逻辑重复** | `project_feasible`（CPU TBB / GPU CUDA）、`compute_Scc`（两套实现）、全局求解（LLT vs Jacobi）各自独立 |
| **数据流不清晰** | copy_mat2thrustvector / copy_thrustvector2mat 分散，sync 时机缺乏统一约定 |
| **模块边界模糊** | 线性代数、约束投影、接触处理散布在多个 .cu/.cuh，无清晰“GPU 后端”模块 |

---

## 2. 重构目标

1. **统一接口**：CPU/GPU 通过同一抽象调用，main 仅依赖 `Solver` 接口。
2. **职责分离**：算法流程与后端实现解耦，Host/Device 数据结构分离。
3. **可维护性**：新增后端（如 Vulkan/Metal）只需实现接口，不改核心流程。
4. **可测试性**：各后端可独立测试，数据拷贝逻辑可单独验证。

---

## 3. 重构架构设计

### 3.1 总体分层

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Application Layer (main.cpp, XPBD_solver)                                   │
│  - 创建 solver，根据 use_GPU 选择 backend                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ADMMCore / ADMMSolver                                                       │
│  - 迭代流程：pre_integration → for admm_it: 碰撞 → 局部步 → 全局步 → 对偶更新  │
│  - 持有 IADMMBackend，通过接口调用各子模块                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  IADMMBackend (抽象接口)                                                      │
│  - ILinearSolver, ILocalProjector, IContactSolver, IConstraintDevice         │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
        ┌───────────┴───────────┐            ┌───────────┴───────────┐
        ▼                       ▼            ▼                       ▼
┌───────────────┐      ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ CPU Backend   │      │ GPU Backend   │  │ (Future)      │  │               │
│ - Eigen LLT   │      │ - Jacobi      │  │ Vulkan/Metal  │  │               │
│ - TBB prox    │      │ - CUDA prox   │  │               │  │               │
│ - ProximalQuery│     │ - BVH_GPU     │  │               │  │               │
└───────────────┘      └───────────────┘  └───────────────┘  └───────────────┘
```

### 3.2 接口定义（伪代码）

```cpp
// solver/core/admm_backend.h

struct ILinearSolver {
    virtual void solve(const VecX& b, VecX& x, int iter_or_factorize) = 0;
    virtual void init_from_assembly(const SpMat& A) = 0;
};

struct ILocalProjector {
    // 弹性约束 prox: z = prox(D*x + Ue)
    virtual void project_elastic() = 0;
};

struct IContactSolver {
    virtual void update_contact_info() = 0;   // 碰撞检测 + 转成 contact 结构
    virtual void compute_Scc() = 0;
    virtual void project_feasible() = 0;      // 速度投影到摩擦锥
};

struct IADMMBackend {
    virtual ILinearSolver* linear_solver() = 0;
    virtual ILocalProjector* local_projector() = 0;
    virtual IContactSolver* contact_solver() = 0;
    virtual void pre_integration(...) = 0;
    virtual void sync_to_host(VecX& x, VecX& v) = 0;  // GPU 需拷贝回 host
};
```

### 3.3 数据流约定

| 阶段 | CPU | GPU |
|------|-----|-----|
| **init** | 在 host 上建 D, M, Wc, 约束；LLT 分解 | 同 host 建 D/M/Wc，再 convert_constraint2device → device |
| **step 开始** | x_curr, v 在 host | x_curr, v 在 device（step 结束前无需 sync） |
| **碰撞** | ProximalQuery 输出 host 接触表 | BVH_GPU 输出 device 接触；convert_DCD_info → ContactDataDevice |
| **局部步** | 直接读 host x，写 host z | 读 device x，写 device z |
| **全局步** | A*x=b，LLT 在 host | A*x=b，Jacobi 在 device |
| **step 结束** | 直接写 m_vertices, m_velocities | sync_to_host → 写 m_vertices, m_velocities |

**原则**：  
- CPU backend 不持有 device 数据。  
- GPU backend 只在 step 边界与 host 交换（init 时 upload，step 后 download）。  
- 避免同一 solver 同时维护两套等价结构。

---

## 4. 目录与模块规划

### 4.1 建议目录结构

```
src/solver/
├── core/                          # 算法流程，无 CUDA 依赖
│   ├── solver.h                   # Solver 基类（可保留）
│   ├── admm_core.h                # ADMMCore：迭代流程
│   ├── admm_backend.h             # IADMMBackend 及各子接口
│   └── admm_core.cpp              # ADMMCore 实现
│
├── backend_cpu/                   # CPU 实现
│   ├── admm_backend_cpu.h
│   ├── admm_backend_cpu.cpp
│   ├── linear_solver_cpu.h/cpp    # Eigen LLT
│   ├── local_projector_cpu.h/cpp  # TBB prox
│   └── contact_solver_cpu.h/cpp   # compute_Scc, project_feasible, ProximalQuery
│
├── backend_gpu/                   # GPU 实现，可放 .cu
│   ├── admm_backend_gpu.h
│   ├── admm_backend_gpu.cu
│   ├── linear_solver_gpu.h/cu     # Jacobi, CuSolverData, CompactSparseMat
│   ├── local_projector_gpu.cuh/cu # 各 *ConstraintDevice::run_proxy
│   ├── contact_solver_gpu.cuh/cu  # ContactDataDevice, compute_Scc_impl_cu, project_impl
│   ├── compact_sparse_matrix_util.cu
│   └── mcuda_wrapper.cu           # pre_integration 等
│
├── shared/                        # CPU/GPU 共用，无 CUDA
│   ├── admm_full_solver.cpp       # 若有共用逻辑（如 sequential_greedy_coloring）
│   └── compact_sparse_matrix.cpp  # Host 端 CompactSparseMat（如需要）
│
├── constraint/                    # 可保留在 constraint/，这里指 solver 对约束的封装
│   └── (现有 triangle/pin/bending/tet device 保持)
│
├── animator.h/cpp                 # 与 backend 无关
├── XPBD_solver.h
└── PBD_solver.h
```

### 4.2 文件迁移映射

| 当前文件 | 目标位置 / 角色 |
|----------|-----------------|
| admm_full_solver.h | 拆为 core/admm_core.h + admm_backend.h |
| admm_full_solver_RL_damping.cpp | backend_cpu/admm_backend_cpu.cpp + contact_solver_cpu |
| admm_full_solver.cpp | shared/ 或 backend_cpu |
| parallel_solver.h | 废弃，逻辑并入 backend_gpu/admm_backend_gpu |
| admm_parallel_global_solver.cu | backend_gpu/admm_backend_gpu.cu |
| jacobi_solver.h/cu | backend_gpu/linear_solver_gpu |
| contact_solver.cu/cuh | backend_gpu/contact_solver_gpu |
| compact_sparse_matrix_util.cu | backend_gpu/ |
| mcuda_wrapper.cu | backend_gpu/ |

---

## 5. 类关系重构

### 5.1 目标继承/组合关系

```
Solver (接口)
  └── ADMMSolver
        ├── backend: std::unique_ptr<IADMMBackend>
        │     ├── ADMMBackendCPU
        │     └── ADMMBackendGPU
        └── 共用参数：dt, admm_max_iter, DCD_interval, mu, warmstart_Ue/Uc, ...
```

- `ADMMSolver` 不再继承 `ADMMSolverFull_RL_damping`。
- `ADMMBackendCPU` 实现 `IADMMBackend`，内部使用 Eigen、TBB、ProximalQuery。
- `ADMMBackendGPU` 实现 `IADMMBackend`，内部使用 CuSolverData、ContactDataDevice、BVH_GPU。

### 5.2 消除 Host/Device 双持有

- **CPU**：只持有 `ADU::SpMatf`, `Gamma_c`, `K_c`, `ProximalQuery::ContactInfoList` 等 host 结构。
- **GPU**：只持有 `CompactSparseMat`, `CuSolverData`, `ContactDataDevice` 等 device 结构。
- **边界**：init 时 CPU 建矩阵 → GPU 通过 `convert_constraint2device` 上传；每帧 step 结束 GPU 通过 `sync_to_host` 下传 x, v。

---

## 6. 分阶段实施计划

### Phase 1：接口与目录（低风险）

1. 定义 `IADMMBackend`, `ILinearSolver`, `IContactSolver`, `ILocalProjector` 接口。
2. 创建 `solver/core/`, `solver/backend_cpu/`, `solver/backend_gpu/` 目录。
3. 将现有 CPU 实现包装成 `ADMMBackendCPU`，保持 `ADMMSolverFull_RL_damping::step` 行为不变。
4. 将现有 GPU 实现包装成 `ADMMBackendGPU`，保持 `ADMMParallelSolver::step` 行为不变。
5. **不修改算法**，只做结构重组。

### Phase 2：统一 contact 接口（中风险）

1. 抽象 `project_feasible` / `compute_Scc` 为 `IContactSolver` 的两个方法。
2. CPU：`ContactSolverCPU` 实现，内部调用现有 `_project_feasible_plain` 和 `compute_Scc`。
3. GPU：`ContactSolverGPU` 实现，内部调用 `project_feasible_parallel` 和 `compute_Scc_parallel`。
4. 确保数值结果与重构前一致（可加单元/回归测试）。

### Phase 3：解耦继承（中高风险）

1. 新建 `ADMMSolver`，持有 `IADMMBackend`，实现 `step()` 为调用 `backend->...`。
2. `ADMMSolver` 从 `Solver` 继承，不再继承 `ADMMSolverFull_RL_damping`。
3. main 中：`use_GPU ? ADMMBackendGPU : ADMMBackendCPU`，注入到 `ADMMSolver`。
4. **（未完成）** 逐步移除 Backend 对旧类的继承：将 `admm_full_solver*.cpp`、`parallel_solver.h`、`admm_parallel_global_solver.cu` 中的逻辑迁移到 `ADMMBackendCPU`/`ADMMBackendGPU` 内部（组合而非继承），使 Backend 仅实现 `IADMMBackend`，不再继承 `ADMMSolverFull_RL_damping`/`ADMMParallelSolver`。

### Phase 4：数据流与 copy 集中（低中风险）

1. 在 `ADMMBackendGPU` 内明确 `upload()` / `download()` 调用点。
2. 将 `copy_mat2thrustvector`、`copy_thrustvector2mat` 收敛到 `backend_gpu/` 内统一封装。
3. 添加注释和断言，确保 sync 时机明确。

### Phase 5：清理与文档（低风险）

1. **删除旧类文件**（前提：Phase 3.4 迁移完成）：删除 `admm_full_solver.h/.cpp`、`admm_full_solver_RL_damping.cpp`、`parallel_solver.h`、`admm_parallel_global_solver.cu` 等；逻辑已迁入 backend_cpu/backend_gpu。
2. 更新 CMake、README、本设计文档。
3. 补充 doxygen 或 markdown 说明各 backend 的职责与数据流。

---

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 数值差异 | 每阶段完成后对比 CPU/GPU 输出与重构前；保留 reference 场景 |
| 构建复杂 | 分阶段提交，每阶段保证可编译、可运行 |
| XPBD 共用 | XPBD 继续使用 `Solver*`，step 内部调用 `solver->step()`，不感知 backend |
| 性能回退 | 每阶段做简单 benchmark（如 armadillo 场景） |

---

## 8. 小结

- **策略模式**：用 `IADMMBackend` 替代深层继承，solver 持有 backend，统一 `step()`。
- **接口收敛**：`project_feasible`、`compute_Scc`、线性求解、局部投影均抽象为接口。
- **数据分离**：CPU 仅 host，GPU 仅 device，边界做显式 copy。
- **模块化**：core / backend_cpu / backend_gpu 职责清晰，便于维护与扩展。

本方案作为讨论与实施参考；实际推进时可按 Phase 1→5 逐步执行，每阶段评审后再进入下一阶段。

---

## 9. Phase 1 & Phase 2 实施记录

### Phase 1 已完成（记录日期：2026-02-21）

1. **接口定义**：`solver/core/admm_backend.h`
   - `IADMMBackend`：precompute(), step()
   - `ILinearSolver`, `IContactSolver`, `ILocalProjector`（占位/部分实现）
2. **目录结构**：`solver/core/`、`solver/backend_cpu/`、`solver/backend_gpu/`
3. **Backend 包装**：`ADMMBackendCPU`、`ADMMBackendGPU`，薄包装原有实现
4. **main 切换**：按 `use_GPU` 创建 `ADMMBackendCPU` 或 `ADMMBackendGPU`
5. **CMake 更新**：加入 backend 子目录的源文件与头文件

### Phase 2 已完成（记录日期：2026-02-21）

1. **IContactSolver 接口**：compute_Scc(bool)、project_feasible(p, contacts, mu, max_iter)
2. **ContactSolverCPU**：`backend_cpu/contact_solver_cpu.h/.cpp`，委托至 `ADMMSolverFull_RL_damping`
3. **ContactSolverGPU**：`backend_gpu/contact_solver_gpu.h/.cu`，委托至 `ADMMParallelSolver`
4. **IADMMBackend::contact_solver()**：返回 `IContactSolver*`，CPU/GPU 各自实现
5. **Step 流程**：尚未改由 contact_solver 调用，逻辑仍保留在 step 内，接口已就绪

### Phase 1&2 阻塞点简要总结

1. **C++ 访问控制**：`ADMMParallelSolver` 的 `step/init/precompute/compute_Scc/project_feasible_parallel` 默认为 private，`ContactSolverGPU` 与 `ADMMBackendGPU` 无法调用，需改为 protected。
2. **前向声明与编译顺序**：`IADMMBackend::contact_solver()` 返回 `IContactSolver*` 时，`IContactSolver` 尚未定义，需前向声明；`ContactSolverGPU` 中 `ADMMParallelSolver` 需完整定义才能调用其方法。
3. **跨编译单元访问**：`ContactSolverGPU` 用 `ADMMParallelSolver*` 调用 protected 方法时 nvcc 报 inaccessible；改为 `ADMMBackendGPU*` 并在 backend 中增加 public wrapper 解决。
4. **头文件依赖**：`ContactSolverCPU` 使用 `ADMMSolverFull_RL_damping*` 但未包含其定义，导致 C2061 等错误。
5. **构建与工具**：CUDA 全量编译耗时，Windows PowerShell 与 bash 语法差异（如 `&&`）需适配。

### Phase 3 已完成（记录日期：2026-02-21）

1. **ADMMSolver 新建**：`solver/core/admm_solver.h/.cpp`
   - 持有 `std::unique_ptr<IADMMBackend>`，不再继承 `ADMMSolverFull_RL_damping`
   - 委托 `init()`, `step()`, `reset()`, `getVertices()`, `getVelocities()` 到 backend
2. **IADMMBackend::as_solver()**：返回 `Solver*`，供 XPBD/main 统一使用
3. **main 更新**：创建 backend → 包装为 `ADMMSolver`，用 `inner_solver()` 做 setup（add_constraints, prox_query 等）
4. **Solver 基类**：`getVertices`, `getVelocities` 改为 virtual

### Phase 4 已完成（记录日期：2026-02-21）

1. **gpu_data_sync.h**：`solver/backend_gpu/gpu_data_sync.h` 记录 Host↔Device 同步约定，re-export `jacobi_solver.h`
2. **sync 注释**：在 `admm_parallel_global_solver.cu` 中标注：
   - init(): UPLOAD x_0, v_0
   - reset(): UPLOAD m_vertices, m_velocities
   - step() animator: UPLOAD x_curr, v_0
   - step() end: DOWNLOAD x_0_device → m_vertices, v_0_device → m_velocities
3. copy 工具仍位于 `jacobi_solver.h/.cu`（可选后续集中到 backend_gpu/）

### Phase 5 已完成（记录日期：2026-02-21）

1. **CMake**：已包含 `solver/core/`、`solver/backend_cpu/`、`solver/backend_gpu/` 源文件
2. **README**：补充 solver 架构与 CPU/GPU 选择说明
3. **设计文档**：更新 Phase 3–5 实施记录

**说明**：Phase 5.1 原未执行，因 Phase 3.4 未完成。已于 2026-02-21 完成 Phase 3.4 与 5.1。

### Phase 3.4 已完成（记录日期：2026-02-21）

1. **ADMMImplCPU / ADMMImplCPUBase**：新建 `solver/backend_cpu/admm_impl_cpu.h/.cpp`
   - 将 `admm_full_solver.cpp`、`admm_full_solver_RL_damping.cpp` 逻辑迁入
   - `ADMMImplCPUBase` 对应原 `ADMMSolverFull`，`ADMMImplCPU` 对应原 `ADMMSolverFull_RL_damping`
2. **ADMMImplGPU**：新建 `solver/backend_gpu/admm_impl_gpu.h/.cu`
   - 将 `parallel_solver.h`、`admm_parallel_global_solver.cu` 逻辑迁入
   - `ADMMImplGPU` 继承 `ADMMImplCPU`，持有 device 结构与 CUDA 逻辑
   - `ContactDataDevice` 迁移至 `admm_impl_gpu.h`
3. **Backend 改用组合**：
   - `ADMMBackendCPU` 持有 `std::unique_ptr<ADMMImplCPU> impl_`，不再继承 `ADMMSolverFull_RL_damping`
   - `ADMMBackendGPU` 持有 `std::unique_ptr<ADMMImplGPU> impl_`，不再继承 `ADMMParallelSolver`
   - `as_solver()` 返回 `impl_.get()`
4. **调用方更新**：`main.cpp`、`solver_config_applier.h`、`contact_solver_cpu`、`XPBD_solver.h` 等改为使用 `ADMMImplCPU*` / `ADMMImplGPU*`
5. **兼容别名**：`admm_full_solver.h` 保留为转发头，`ADMMSolverFull_RL_damping` = `ADMMImplCPU`

### Phase 5.1 已完成（记录日期：2026-02-21）

1. **删除旧实现文件**：
   - `admm_full_solver.cpp`
   - `admm_full_solver_RL_damping.cpp`
   - `parallel_solver.h`
   - `admm_parallel_global_solver.cu`
2. **admm_full_solver.h**：改为兼容转发头，仅提供类型别名并 include 新头文件
3. **引用更新**：`contact_solver.cu` 改为 include `admm_impl_gpu.h`；`collision_detection_cuda.cpp` 改为 include `solver.h`

### 依赖倒置（Dependency Inversion，2026-02-21）

**目的**：减少 main 对 backend/impl 头文件的直接依赖，应用层仅依赖抽象接口和轻量 facade。

**新增文件**：
- `solver/admm_solver_factory.h/.cpp`：`create_admm_backend(bool use_gpu)`，按 flag 创建 CPU 或 GPU backend
- `mediator/solver_config_applier.h/.cpp`：`apply_solver_config(Solver*, ...)`、`setup_parallel_backend(Solver*, ...)`，接受 `Solver*`，内部做 impl 类型 cast
- `mediator/contact_display_helper.h/.cpp`：`update_contact_display(ADMMSolver*, bool use_gpu, PointCloud*)`，统一接触可视化，隐藏 impl 类型

**main.cpp 变更**：
- 移除 include：`admm_impl_cpu.h`、`admm_impl_gpu.h`、`admm_backend_cpu.h`、`admm_backend_gpu.h`
- 新增 include：`admm_solver_factory.h`、`contact_display_helper.h`
- 使用 `create_admm_backend()` 创建 backend，`apply_solver_config()` / `setup_parallel_backend()` 做配置，`update_contact_display()` 做接触显示

### 类型别名移除（2026-02-21）

- 移除 `ADMMSolverFull_RL_damping`、`ADMMSolverFull` 别名
- 删除 `admm_full_solver.h`（不再保留转发头）
- `XPBD_solver.h`、`twobody.cpp`、`triangle.cpp`、`tetrahedral.cpp` 直接使用 `ADMMImplCPU`

### 遵守 refactor-guidelines.md

- **组合优于继承**：Backend 通过 `std::unique_ptr<ADMMImpl*> impl_` 持有实现，不再继承旧类
- **通过接口暴露**：`IADMMBackend` 提供 `as_solver()`，外部仅依赖接口
- **前向声明**：`contact_solver_gpu.h` 前向声明 `ADMMBackendGPU`，减少头文件依赖
- **增量替换**：先建新 impl 再迁逻辑，保持可编译；删除旧文件前完成迁移
- **Sync 标注**：GPU impl 中保留 `[Phase 4] UPLOAD/DOWNLOAD` 等注释

---

## 10. 后续工作（未完成）

| 阶段 | 工作 | 当前状态 |
|------|------|----------|
| - | Phase 3.4、5.1 已完成 | ✓ |
| - | 依赖倒置（factory + config/contact helper）已完成 | ✓ |
| - | 类型别名移除、admm_full_solver.h 删除已完成 | ✓ |
