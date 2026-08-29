# 仿真执行链路 (Simulation Execution Chain)

基于 `agent_aux` 记忆与代码解析整理的完整仿真执行流程。

---

## 1. 程序入口与初始化

### 1.1 main() 流程 (`src/main.cpp`)

```
main(argc, argv)
  ├─ 解析参数: config_file_path, resource_file_path (argparse)
  ├─ app_config.load_config(config_path)     // TOML 场景配置
  ├─ 创建输出目录 out_path
  ├─ 加载网格 (mesh)
  │    └─ 按 app_config.meshes: tri/tet/static，add_mesh → center/scale/rotate/translate，material
  ├─ 静态网格: static_mesh_id_begin, static_vert_begin, precompute_geometry_info
  ├─ 约束构建 (Mesh2Constraint)
  │    └─ geometry_to_constraints(*mesh, mesh_id, material, cslist, XPBD_cslist, tid=1,2,3)
  │         tid=1: TRI → TriangleConstraint (stretch)
  │         tid=2: TET → TetrahedralConstraint
  │         tid=3: TRI → BendingConstraint
  ├─ Pin: addPins(pin_ids), pin_to_constraints → cslist, pin_start/pin_end
  ├─ 创建求解器 (CPU 或 GPU)
  │    ├─ use_GPU → ADMMParallelSolver
  │    └─ else  → ADMMSolverFull_RL_damping
  ├─ 设置求解器参数 (admm_max_iter, DCD_interval, warmstart, contact 参数, damp, 等)
  ├─ add_constraints(cslist), add_XPBDConstraints(XPBD_cslist)
  ├─ use_GPU 时: convert_constraint2device()
  ├─ 碰撞/接触
  │    ├─ ContactParameter: broadphase_radius, thickness
  │    ├─ prox_query = ProximalQuery(mesh, max_collision)
  │    ├─ broad_phase = BroadPhase_embree(REBUILD/REFIT), build()
  │    └─ thickness_validation_check(mesh)
  ├─ 障碍物: addObstacle(PlaneObstacle(...))
  ├─ 动画: 若有 mesh_animate_info → ScriptAnimator, attach_animate2mesh
  ├─ solver->init(verts, mass, dt)
  ├─ XPBD 若启用: XPBDSolver::init(solver, XPBD_iter, sub_step)
  ├─ BVH_GPU: app.bvh 与 solver->bvh 互相挂接，bvh->init(), construct(), update_bvh_draw()
  └─ polyscope::init(), registerSurfaceMesh/PointCloud/CurveNetwork, userCallback = main_loop, polyscope::show()
```

### 1.2 配置来源 (`mediator/toml_to_config.h`)

- **APPConfig**: scene_name, out_file, use_GPU, end_frame, show_windows
- **Global**: dt, pin_ids, g, sub_step
- **Solver.ADMM**: admm_max_iter, DCD_interval, GS_max_iter, Global_Jacobi_iter, warmstart_Uc/Ue
- **Solver.Contact**: enable, use_CCD, w_scale, kappa, beta, mu, unique, use_jacobi, use_heu_wc
- **Solver.Damp**: kl, km
- **DCD**: narrow (max_collision, thickness), broad (radius, strategy)
- **Mesh**: path, type (tri/tet/static), density, transform, material (k), animate_path

### 1.3 约束到求解器 (`mediator/mesh_to_constraint.h`)

- **Mesh2Constraint::geometry_to_constraints**: 按 typeID 1/2/3 生成 TriangleConstraint、TetrahedralConstraint、BendingConstraint，写入 cslist/XPBD_cslist，并推进 `curr_start_row`。
- **Mesh2Constraint::pin_to_constraints**: 根据 pin_ids 生成 PinConstraint，写入 cslist，并填充 pin_v、pin_start、pin_end。

---

## 2. 每帧主循环：main_loop()

```
main_loop()  // polyscope 每帧回调
  ├─ UI: pause, step, reset, export obj
  ├─ if reset → solver->reset(verts), step_idx=0, pause=true
  ├─ if !pause || step:
  │    ├─ 若 enable_XPBD && XPBD_solver → XPBD_solver->step()
  │    └─ 否则 → for sub_step: app.solver->step()
  │    ├─ update_bvh_draw()
  │    ├─ 若 export_obj && save_res → 按 mesh 或整体写 OBJ/PLY
  │    ├─ 若 show_windows: 更新 velocity 显示、end_of_step_callback
  │    └─ step_idx++
  ├─ 更新可视化: vis_meshP->updateVertexPositions, vis_contactP (contact points/normals), vis_bvh
  └─ if step_idx > end_frame → pause, polyscope::unshow()
```

---

## 3. ADMM 单步：solver->step()

### 3.1 CPU 路径：ADMMSolverFull_RL_damping::step()

入口：`step()` → `step_fast()`。

```
step_fast()
  ├─ 稳定性系数: epsilon, gamma (kappa, beta, dt)
  ├─ x_0 = m_vertices, v_0 = m_velocities
  ├─ 显式积分: v_0 重力 (pin 处清零), x_tilde = x_0 + dt*v_0, M_x_tilde = M * x_tilde
  ├─ 初始化: x_curr = x_tilde (动态), x_curr(静态部分) = x_0, z=0, DX=0
  ├─ 若 animator: animate_all(verts, x_curr, v_0, dt)
  ├─ b_ini = Damp_Mat * x_0, p = v_0
  ├─ 可选: warmstart_Ue/Uc 不清零
  └─ for admm_it = 0 .. admm_max_iter:
       ├─ 【碰撞】若 enable_frictional_contact && (admm_it % DCD_interval == 0):
       │    ├─ use_CCD ? prox_query->proximal_query_with_CCD(x_0, x_curr) : prox_query->proximal_query(x_curr)
       │    ├─ 若 use_unique_contact: prox_query->unique_contact()
       │    └─ need_recompute_Scc = true, find_contact_islands()
       ├─ 【局部步】并行:
       │    ├─ 弹性: DX = D*x_curr, z = DX + Ue; 对每个 constraint: ct->prox(zi), 写回 z
       │    └─ 接触: p = (x_curr - x_0 + Uc)/dt; 若 need_recompute_Scc: compute_Scc(); project_feasible(p, contacts, mu, gs_max_iter)
       ├─ 【全局步】b = M_x_tilde + dt²*D'*We'*We*(z - Ue) + dt²*Wc*(dt*p + x_0 - Uc) + b_ini
       │            x_curr(动态) = LLT.solve(b)  (三列并行)
       ├─ 对偶更新: DX = D*x_curr; Ue += DX - z; Uc += x_curr - x_0 - p*dt
       └─ (可选) 残差打印
  └─ 收尾: m_velocities = (x_curr - x_0)/dt, m_vertices = x_curr, step_cnt++
```

- **compute_Scc**: 按 contact_info_list 算 Gamma_c, K_c（与 contact_w_list / M_inv 相关），用于接触投影。
- **project_feasible**: 将速度 p 投影到摩擦锥内（_project_feasible_plain，PGS 风格迭代）。

### 3.2 GPU 路径：ADMMParallelSolver::step()

结构与 CPU 一致，但数据与运算在设备上：

```
ADMMParallelSolver::step()
  ├─ epsilon, gamma 同 CPU
  ├─ 若 animator: animate_all → 拷贝 x_curr, v_0 到 device
  ├─ do_pre_integration(...) 在 GPU 上算 x_curr, M_x_tilde 等
  ├─ warmstart_Ue/Uc 可选清零，p_device = v_0
  └─ for admm_it:
       ├─ 【碰撞】若 enable_frictional_contact && (admm_it % DCD_interval == 0):
       │    ├─ bvh->update(x_curr_device, ...), bvh->dcd()
       │    └─ convert_DCD_info(bvh, x_curr_device, ct_data_device), need_recompute_Scc = true
       ├─ 【局部步】
       │    ├─ 弹性: op_Ax(D_device, x_curr, DX); z = DX + Ue; 各 constraint device (tet/triangle/bending/pin)->run_proxy(z 对应块)
       │    └─ 接触: p = (x_curr - x_0 + Uc)/dt; compute_Scc_parallel(); project_feasible_parallel()
       ├─ 【全局步】b_curr = M_x_tilde + dt²*D'*We'*We*(z - Ue) + dt²*Wc*(dt*p + x_0 - Uc); 未加 damp 时用 Jacobi_global(Global_Jacobi_iter) 解 A*x_curr = b_curr
       └─ 对偶更新在设备上，最后拷贝 x_curr 回 host，更新 m_vertices, m_velocities
```

- **BVH_GPU**: init → construct；每步 `update(verts)` → `dcd()`，结果通过 `convert_DCD_info` 转成 ContactDataDevice 供 contact 子问题使用。
- **convert_constraint2device**: 将 D、dt²*D'*We'*We、M、dt²*Wc 等转为设备上的 CompactSparseMat，约束转为各 *ConstraintDevice。

---

## 4. 碰撞检测链路

### 4.1 CPU：ProximalQuery (Embree BVH + 自研 narrow)

- **broad_phase**: `BroadPhase_embree`，策略 REBUILD/REFIT。
- **proximal_query(pos)**:
  1. `broad_phase->update(pos)` 更新 BVH
  2. `broad_phase->query_point_triangle(pos, candidates_pt)`
  3. `broad_phase->query_edge_edge(pos, candidates_ee)`
  4. `query_point_triangle(pos)` → 对 candidates_pt 做 narrow_PT，写入 contact_info_list
  5. `query_edge_edge(pos)` → 对 candidates_ee 做 narrow_EE，写入 contact_info_list
- **proximal_query_with_CCD(pos_t0, pos)**: 同上，但 narrow 用带 CCD 的版本（若 USE_CCD）。
- **unique_contact()**: 去重接触对。

### 4.2 GPU：BVH_GPU (LBVH)

- **BVH_GPU**: 使用 `lbvh_f`（面）、`lbvh_e`（边），数据在 device（d_verts, d_faces, d_edges, d_surfVertIdx, d_collisonPairs, d_cpNum, d_contact_info）。
- **update(verts, ...)**: 更新顶点，可能重建/refit LBVH。
- **dcd()**: 在 GPU 上执行面-面/边-边碰撞检测，写出 d_collisonPairs, d_contact_info, d_cpNum。
- **convert_DCD_info(bvh, x_curr_device, ct_data_device)**: 将 GPU 碰撞结果转为 ContactDataDevice（bary, normal, point, h_cN, pair_type, 等），供 compute_Scc_parallel 与 project_feasible_parallel 使用。

---

## 5. XPBD 分支（enable_XPBD）

**XPBDSolver::step()** → **step_XPBD()**:

- 每帧内按 **subSteps** 做子步；子步内 h = dt / subSteps。
- 每子步:
  1. 重力更新速度，x_curr += v * h。
  2. 内层迭代 (cur_iter < m_maxIterations):
     - 若启用接触且 cur_iter % DCD_interval == 0: 与 ADMM 相同调用 prox_query->proximal_query（或 CCD 版）+ unique_contact，need_recompute_Scc = true。
     - 对 m_XPBDconstraints: updateConstraint(); solvePositionConstraint(x_curr, ..., M_inv, cur_iter, h)。
     - 若 enable_frictional_contact: v_curr = (x_curr - x_prev)/h；若 need_recompute_Scc 则 compute_Scc(true)；project_feasible(v_curr, ...)；x_curr = x_prev + h*v_curr。
  3. v_curr = (x_curr - x_prev)/h，并做阻尼（如 *= 0.9995）。
- 最后写回 solver->m_vertices, solver->m_velocities。

---

## 6. 数据流小结

| 阶段           | CPU 路径                         | GPU 路径                          |
|----------------|----------------------------------|-----------------------------------|
| 场景与约束     | TOML → APPConfig；Mesh2Constraint → cslist/XPBD_cslist | 同左；再 convert_constraint2device |
| 每帧驱动       | main_loop → solver->step() 或 XPBD_solver->step() | 同左                               |
| 碰撞           | ProximalQuery (Embree + narrow_phase) | BVH_GPU (LBVH) + convert_DCD_info   |
| 弹性局部       | D*x → z; 各 Constraint::prox(z) | D_device*x → z; *ConstraintDevice::run_proxy(z) |
| 接触局部       | compute_Scc + project_feasible   | compute_Scc_parallel + project_feasible_parallel |
| 全局           | (M + dt²*D'*We'*We*D + dt²*Wc + Damp)*x = b, LLT 解 | 同构，Jacobi_global 迭代解          |
| 状态输出       | m_vertices, m_velocities         | device 拷回 host 再写回 m_vertices/m_velocities |

---

## 7. 关键文件索引

- **入口与循环**: `src/main.cpp` (main, main_loop)
- **配置**: `src/mediator/toml_to_config.h` (APPConfig, load_config)
- **约束构建**: `src/mediator/mesh_to_constraint.h`
- **求解器基类**: `src/solver/solver.h` (Solver)
- **ADMM 全量**: `src/solver/admm_full_solver.h`, `src/solver/admm_full_solver_RL_damping.cpp` (step_fast, compute_Scc, project_feasible)
- **ADMM 并行**: `src/solver/parallel_solver.h`, `src/solver/admm_parallel_global_solver.cpp` (step, Jacobi_global, convert_constraint2device)
- **XPBD**: `src/solver/XPBD_solver.h` (step_XPBD)
- **碰撞 CPU**: `src/contact/narrow_phase.h`, `narrow_phase.cpp` (ProximalQuery); `broad_phase.h`, `broad_phase_embree.cpp`
- **碰撞 GPU**: `src/contact/collision_detection_cuda.h`, `collision_detection_cuda.cpp` (BVH_GPU, dcd, convert_DCD_info)
- **接触求解器**: `src/solver/contact_solver.cu` (project_feasible_parallel 等设备侧实现)

以上为当前代码库下的完整仿真执行链路；后续若增删模块或切换 CPU/GPU/XPBD，可据此快速定位调用关系。
