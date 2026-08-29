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

---

## Phase C：效率优化（完成，默认行为不变）

1. **矩阵导出加开关**：`[solver.admm] dump_A = true`（默认 false）才写 `m_A_damp` 到磁盘。
2. **CPU step_fast 缓冲复用**：x_0/v_0/x_tilde/M_x_tilde/b/b_ini/x_curr/z/DX/p 移为
   `ADMMImplCPUBase` 成员，避免每步 Eigen 分配；GPU 侧删除重复的 host 成员
   （x_curr/p/b_curr，其中 b_curr 本就未被使用）。
3. **GPU 死 LLT 移除**：GPU precompute 的 `m_A_damp` 装配与 `SimplicialLLT` 分解
   从未被 GPU step（Jacobi）使用，删除。
4. **GPU headless 免同步**：`ADMMSolver::set_sync_to_host(false)` 时 GPU step 跳过
   每步 device→host 下载；app 在无窗口且无 animator 时自动关闭。
   （注意：animator 依赖 host 数据，故有 animator 时保持同步。）
5. **coloring 并行接触投影（opt-in）**：`[solver.contact] coloring = true` 时 CPU
   接触 GS 按贪心染色并行（颜色顺序执行、颜色内并行）；单接触更新提取为
   `_project_one_contact` 与串行路径共享；未染色溢出接触保持串行兜底。
   默认关闭 → 数值与原来完全一致。

### 未做（记录原因）
- 多 RHS LLT：现状为 3 列并行单列 solve（3 线程），Eigen 稀疏三角求解本身不并行，
  并行 3 列已是合理方案；多 RHS 单次 solve 反而串行，未改。
- `use_jacobi`、`GS_global`、`find_contact_islands`、`compute_SaSb`：死代码/空桩，
  保留待清理（见 Phase D 记录）。
- GPU 阻尼缺失疑点：GPU step 无 `b_ini`（`m_Damp_Mat * x_0`）项，与 CPU 行为差异
  需在运行时验证后处理。

---

## Phase D：模块化拆散（完成）+ 清理候选

### D1. mediator 拆散（module-modularization-design.md 全 Phase 完成）

| 原路径 | 新路径 |
|--------|--------|
| mediator/toml_to_config.h | **config/app_config.h** |
| mediator/mesh_from_config.h | **config/mesh_from_config.h** |
| mediator/mesh_to_constraint.h | **constraint/mesh_to_constraint.h** |
| mediator/solver_config_applier.h/.cpp | **solver/solver_config_applier.h/.cpp** |
| mediator/DCD_validation_check.h | **contact/dcd_validation.h** |
| mediator/bvh_display_helper.h | **contact/bvh_display.h** |
| mediator/contact_display_helper.h/.cpp | **contact/contact_display.h/.cpp** |
| mediator/export_frame.h | **mesh/export_frame.h**（并移除对 config 的不必要 include） |
| mediator/app_context.h | **app/app_context.h** |
| mediator/app_initializer.h/.cpp | **app/app_initializer.h/.cpp** |
| mediator/toml_to_scene.h | 删除（空文件） |

CMake GLOB 同步更新（src/app、src/config 加入，src/mediator 移除）。
`src/mediator/` 目录已不存在。

### D2. 清理候选（未做，低优先级）
- `solver/PBD_solver.h`：仅含注释掉的类，已无任何引用 → 可归档/删除。
- `solver/backend_gpu/admm_impl_gpu.h` 中 `GS_global` 声明+实现未被调用 → 可删。
- `use_jacobi`（config/impl 成员）只写不读 → 可删（保留 toml 兼容解析）。
- `mutils/helper_cuda.h`、`helper_string.h`、`constraint/spring_constraint.h`：
  未被 main 依赖链引用（见 src-unused-files-and-zensim.md）。
- `find_contact_islands()`/`compute_SaSb()` 空桩；collision_detection_cuda.cpp
  两处 "TODO init got memory leak!" 注释待核实。
- twobody/triangle/tetrahedral.cpp：可选 target 已注释且代码已过期（不参与构建）。

---

## Phase D 追加：子问题接口真实化 + 约束工厂 + prox 视图化（完成）

### D3. ILinearSolver / ILocalProjector 真实实现
- `solver/core/admm_backend.h`：占位接口替换为真实接口
  - `ILinearSolver::solve()` — 全局步：CPU=SimplicialLLT（3 列并行），GPU=Jacobi
  - `ILocalProjector::project_elastic()` — 局部步：z = prox(D*x + Ue)
  - `IADMMBackend::linear_solver() / local_projector()` 暴露组件
- 具体实现：
  - `backend_cpu/linear_solver_cpu.h`、`backend_cpu/local_projector_cpu.h`
  - `backend_gpu/linear_solver_gpu.h`、`backend_gpu/local_projector_gpu.h`
    （通过 `ADMMImplGPU::do_jacobi_global()` / `do_project_elastic()` 包装调用设备逻辑）
- 归属：`ADMMImplCPUBase` 持有 `unique_ptr<ILinearSolver/ILocalProjector>`，
  CPU/GPU impl 的构造函数各自装入对应实现；step 热循环经由接口调用（行为不变）。
- 意义：全局求解器（未来 CG/批处理 LDLT）与局部投影可插拔替换，无需改 step。

### D4. 约束工厂（typeID → creator）
- 新增 `constraint/constraint_factory.h`：`register_constraint_creator(type_id, fn)`
  / `create_constraints_for_type(...)` 注册表。
- `mesh_to_constraint.h` 的内建 tri/tet/bending 创建逻辑迁入
  `ensure_builtin_constraints_registered()`（惰性注册）；`geometry_to_constraints`
  改为走注册表。新增约束类型不再需要改 mesh_to_constraint。

### D5. prox 按类型分块 dispatch（CPU 局部步）
- `ADMMImplCPUBase::constraint_type_ranges`：precompute 时按类型记录连续区间；
  局部步对每个类型区间各发一个 `parallel_for`（GPU 侧原本就是按类型 run_proxy）。
- 尝试过把 `Constraint::prox` 改为 `Eigen::Ref<Matf_XX,0,Stride<D,D>>` 直接绑定
  `z.block(...)` 以消除每约束拷贝——Eigen 的 Ref 无法绑定列优先矩阵的行块
  （inner stride ≠ 1，编译期拒绝），已回退；拷贝开销本身很小（每约束 12–36 floats），
  维持 `Matf_XX&` 签名。
- 与局部步/全局步一起被封装为 `ILocalProjector` / `ILinearSolver`（见 D3）。

### 验证
- 全部阶段构建通过（Release, MSVC + nvcc）。
- 数值行为：默认配置下（coloring/dump_A 均关）与重构前一致；prox 视图化仅改变
  数据通路（同一批浮点运算），coloring 为 opt-in。
