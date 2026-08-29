# 重构 Phase A/B 进度记录（2026-03，后续轮次）

## Phase A：碰撞检测接口化（完成）

- 新增 `contact/icollision_detector.h`：`ADU::ICollisionDetector` 宿主侧接口
  （detect / contact_count / max_slots / peak_contact_count / points / normals / clear）。
- `ProximalQuery`（CPU）与 `BVH_GPU`（GPU）实现该接口；
  GPU 的 device 快速路径（`update(Real*)` 等）保留在具体类上，避免宿主往返。
- `contact_display_helper` 改为接收 `ICollisionDetector*`，**删除 impl dynamic_cast**；
  main.cpp 的接触显示与 max-collision 统计统一走接口。
- 注意：接口方法命名 `points()/normals()` 是为了避免与
  `ProximalQuery::contact_points/contact_normals` 数据成员同名冲突。

## Phase B：配置类型化 + 统一工厂 + Solver 瘦身 + XPBD 归档（完成）

### B1. XPBD 归档（不进入生效代码）
- 文件移至 `archive/xpbd/`（XPBD_solver.h、XPBD_constraints.h、XPBD_*.cpp），不参与编译。
- `XPBD_utils` 保留在 `src/constraint/xpbd_utils.h`（ADMM triangle constraint 使用
  `get_FEMTriangleGradient`）。
- 生效路径清理：solver.h / constraint.h / mesh_to_constraint.h / app_initializer /
  app_context / main.cpp / collision_detection_cuda.cpp；`compute_Scc(is_XPBD)` 参数删除；
  `[xpbd]` TOML 段保留解析、标记 DEPRECATED。
- 详见 `agent_aux/xpbd-archive.md`。

### B2. 配置类型化（消灭 dynamic_cast）
- 新增 `solver/core/admm_config.h`：`ADMMSolverConfig` 纯数据配置结构。
- `solver_config_applier` 重写为 `build_solver_config(cfg, ctx) -> ADMMSolverConfig`，
  不再触碰任何 impl 类型。
- `IADMMBackend` 新增 `apply_config() / finalize_constraints() / set_constraint_dim()`；
  `ADMMSolver` facade 提供同名转发，应用层零 cast。

### B3. 统一工厂
- `solver/admm_solver_factory.h`：新增 `enum class SolverType { ADMM_CPU, ADMM_GPU }`
  与 `create_solver(SolverType)`；未来 solver 直接加枚举项。

### B4. Solver 基类瘦身
- `admm_max_iter`、`m_nCDim`、`parallel` 移入 `ADMMImplCPUBase`（ADMM 专属）。
- 删除死成员：`m_vStatic`、`m_M_bar_vec`、`m_M_bar_inv_vec`。
- `addPins/add_constraints/add_nodal_constraints/addObstacle` 改 virtual，
  `ADMMSolver` 转发到 inner solver（消除 facade 自身的状态分裂陷阱）。

### B5. 修复重构引入的两个回归（重要）
1. **convert_constraint2device 时序**：原代码在 `add_constraints` 之后调用；
   重构后跑到约束注册之前 → GPU 设备约束为空。已修复：
   `set_constraint_dim` + `finalize_constraints()` 在约束装配完成后调用。
2. **bvh 指针挂错对象**：重构前 `app.solver` 就是 impl，`app.solver->bvh = &bvh`
   直接写入 impl；重构后写到了 facade 自己的成员（impl 的 bvh 恒为 null，
   GPU step 会解引用空指针）。已修复：`inner->bvh = &bvh; bvh.solver = inner;`。

## 遗留观察（Phase C 候选）

- GPU precompute 里 `m_LLT_solver->compute(m_A_damp)` 从未被 GPU step 使用
  （step 只用 Jacobi）——死开销，可删（连带 `m_A_damp`/`m_Damp_Mat` 装配可评估）。
- CPU precompute 每次把 `m_A_damp` 写磁盘（`saveMarket`，根目录 `m_A_damp_MD_*` 来源）→ 加开关。
- GPU step 无 b_ini（阻尼）项：CPU 有 `b_ini = m_Damp_Mat * x_0`，
  GPU 是否应用阻尼需核实（行为差异风险）。
- CPU `_project_feasible_plain` 全串行 GS；`sequential_greedy_coloring` 已实现但被
  硬编码关闭（`coloring_parallel_contact = false`），`find_contact_islands()` 为空。
