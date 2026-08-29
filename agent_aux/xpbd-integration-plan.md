# XPBD 仿真器集成规划（parallel_admm）

**日期**: 2026-03
**目标**: 在本框架中实现独立的 XPBD 求解器（第二个 solver），可运行论文对比场景
（`test_XPBD_ADMM.toml`、`armadillo_xpbd*.toml` 等），复用重构后的基础设施。

---

## 1. 调研：GitHub 代表性 XPBD 实现

| 参考 | 要点 | 链接 |
|------|------|------|
| **Macklin et al. 2016 XPBD 论文** | 理论源头：约束函数 C = sqrt(2U') 的 compliant XPBD；本仓库 archive/xpbd 与 xpbd_utils.h 的 FEM 数学即此风格 | matthias-research.github.io/pages |
| **InteractiveComputerGraphics/PositionBasedDynamics**（Bender 组） | 教科书级 C++ 库：XPBD 求解循环、Distance/Strain(3D/2D)/Bending 约束、四面体/三角形网格、接触（PT/EE）与摩擦；约束为数据驱动、无每约束虚调用热点 | https://github.com/InteractiveComputerGraphics/PositionBasedDynamics |
| **Blender geometry/xpbd 模块** | 生产级集成（4.x 布料/软体）：`GEO_xpbd_constraint_*` 约束类 + 碰撞面约束；求解循环与数据布局值得参考 | https://github.com/blender/blender/tree/main/source/blender/geometry/xpbd |
| **WilKam01/GPU_Soft_Body_Physics** | 学士论文：XPBD + 四面体变形软体（GPU），结构简单清晰 | https://github.com/WilKam01/GPU_Soft_Body_Physics |
| **Q-Minh/position-based-dynamics** | XPBD 软体 + 虚拟切割，含 FEM tet 约束 | https://github.com/Q-Minh/position-based-dynamics |
| **vitalight/Velvet** | CUDA XPBD 布料引擎（GPU 路线参考，本期不做） | https://github.com/vitalight/Velvet |

**结论**：约束数学采用 Macklin 风格（本仓库已有，与论文一致）；求解循环/数据结构参考
Bender 库与 Blender 模块（类型分块、无虚调用热点、SoA 约束数据）。

---

## 2. 框架现状：可复用资产

| 资产 | 位置 | XPBD 的用法 |
|------|------|-------------|
| `Solver` 基类（verts/vel/mass/M_inv/pins/dt/obstacles/animator + init/reset/step/getVertices/getVelocities 虚接口） | `solver/solver.h` | XPBDSolver 直接继承；显示/导出/重置全复用 |
| `ICollisionDetector`（ProximalQuery CPU / BVH_GPU GPU） | `contact/icollision_detector.h` | XPBD 碰撞检测走 ProximalQuery（CPU 期）；BVH_GPU 为未来扩展 |
| `XPBD_utils` 数学（PT/EE 距离、FEM 梯度） | `src/constraint/xpbd_utils.h` | 接触约束与 FEM 约束的数学核心 |
| 归档 XPBD 约束类（FEMTriangle/XPBDBending/XPBD_FEMTet + 实现 cpp） | `archive/xpbd/` | 复活为独立约束模块（见 §4） |
| 约束工厂注册表（ADU::Constraint 体系） | `constraint/constraint_factory.h` | 设计对称的 XPBD 约束工厂（XPBDConstraint 体系） |
| 统一工厂 `create_solver(SolverType)` | `solver/admm_solver_factory.h` | 加 `SolverType::XPBD` |
| TOML 配置体系（APPConfig + 纯数据 Config 结构） | `config/app_config.h`、`solver/core/admm_config.h` | 新建 `XPBDConfig`；`[xpbd]` 段已在解析（DEPRECATED 注释），扩展字段 |
| `MeshData`/`PhyxMaterial`/`Mesh2Constraint` | `mesh/`、`constraint/mesh_to_constraint.h` | 网格几何（faces/tets/adjacency/质量）与材质参数（stretch/bending stiffness） |

---

## 3. 目标架构

```
Solver（通用状态 + 虚接口）
 ├── ADMMImplCPU / ADMMImplGPU         （现有）
 └── XPBDSolver : Solver               （新，独立实现）
       ├── 状态：继承 Solver（不复制 ADMM 任何成员）
       ├── 配置：XPBDConfig（substeps / iterations / stiffness / contact）
       ├── 碰撞：ICollisionDetector*（app 注入 ProximalQuery）
       ├── 约束：XPBDConstraint 体系（数据驱动，类型分块）
       │     ├── FEMTriangleConstraint（布料拉伸，2D FEM，Macklin 公式）
       │     ├── XPBD_FEMTetConstraint（软体，3D FEM）
       │     ├── XPBDBendingConstraint（二次形弯曲）
       │     └── 接触：PT/EE 距离约束 + 摩擦锥（XPBD_utils 数学）
       └── step()：substep 循环
             对每个 substep：
               v += g*dt（pin 跳过）；x += v*dt
               for iter in 1..max_iter:
                 for 弹性约束块: solvePositionConstraint(x, inv_mass, dt, lambda)
                 if 需要碰撞: detector->detect(x) → 接触列表
                 for 接触约束块: solve + 摩擦投影
               v = (x - x_prev)/dt（可选 damping）
```

**装配链（app_initializer）**：
```
TOML: [solver] type = "admm_cpu" | "admm_gpu" | "xpbd"
      [xpbd] substeps / iter / stiffness / contact_stiffness / friction / damping ...
→ create_solver(SolverType::XPBD) → XPBDSolver
→ solver->apply_config(XPBDConfig)（仿 ADMMSolverConfig 纯数据）
→ 构建 XPBD 约束（XPBD 约束工厂，Mesh2Constraint 增加 XPBD 路径或对称新函数）
→ inner 装配：pins / obstacles / animator / prox_query（现有流程共用）
→ solver->init(...) / step() / getVertices()（现有 main 循环零改动）
```

**显示/导出**：main.cpp 走 `Solver*` 虚接口，无需改动；接触显示走
`ICollisionDetector`（XPBD 用 ProximalQuery 时自动工作）。

---

## 4. 关键设计决策

1. **独立类，不寄生 ADMM**：`XPBDSolver : Solver`，自己的 step() 与约束容器；
   绝不 reinterpret_cast 到 ADMMImplCPU（归档前旧实现的病根）。
2. **复活 + 重构约束**：`archive/xpbd/XPBD_constraints.h` 的 FEM 数学（Macklin 风格，
   与论文一致）移回 `src/constraint/xpbd/`；但按 Bender/Blender 风格重构：
   - 数据驱动（约束数据与求解函数分离，避免每约束虚调用——与 ADMM 侧
     `constraint_type_ranges` 分块 dispatch 的思路一致）
   - `XPBD_utils` 保持共享（已在 `src/constraint/xpbd_utils.h`）
3. **接触**：检测用 `ICollisionDetector`（ProximalQuery → ContactInfoList），
   解算用 XPBD 距离约束（XPBD_utils::solve_TrianglePointDistance/EdgeEdge，
   压缩/拉伸刚度）+ 摩擦锥投影（参考旧 solve_contact_constraints 与 Blender
   collision_face 约束）。接触刚度来自场景 `contact.w_scale`（与旧版一致）。
4. **配置**：新建 `XPBDConfig` 纯数据（substeps、max_iter、stiffness 缩放、
   contact_stiffness、mu、thickness、damping、use_CCD…），`[xpbd]` TOML 段扩展
   （现有 enable/iter 保留解析，缺省值默认，旧场景文件不破坏）；
   `[solver] type` 新增 `"xpbd"` 选择（缺省 admm 行为不变）。
5. **工厂**：`SolverType::XPBD` 加入 `create_solver()`；app_initializer 按 type 分支
   装配（ADMM 现有路径不动）。
6. **GPU**：第一期碰撞与解算均在 CPU（ProximalQuery）；BVH_GPU + CUDA 接触解算
   作为后续扩展（参考 Velvet），不在本期范围。

---

## 5. 实现阶段（每阶段可编译、可运行）

| 阶段 | 内容 | 验收 |
|------|------|------|
| **P1 约束复活** | 将 archive/xpbd 的 XPBDConstraint 类移回 `src/constraint/xpbd/`，重构为数据驱动 + 类型分块；XPBD 约束工厂（对称 constraint_factory） | 编译通过；单元级手工验证 FEM 梯度 |
| **P2 求解器骨架** | `XPBDSolver : Solver`：substep 循环（重力/速度/位置），无约束 step；`XPBDConfig` + toml 解析；`SolverType::XPBD` + `create_solver`；app_initializer 装配分支（pins/obstacles/animator/prox_query 共用） | `[solver] type="xpbd"` 场景跑通（自由落体） |
| **P3 弹性约束** | 接入 FEMTriangle（布料）/XPBD_FEMTet（软体）/Bending；材质参数（PhyxMaterial）→ 约束刚度映射 | `rod_twist`、布料场景出稳定形变 |
| **P4 碰撞与摩擦** | ProximalQuery 检测 → PT/EE 距离约束 + 摩擦锥；contact 显示/统计复用 | `test_XPBD_ADMM.toml`、`armadillo_xpbd*.toml` 跑通 |
| **P5 验证与收尾** | 与归档前旧实现/论文结果对比（几何轨迹、能量）；清理；文档更新（Readme、agent_aux） | 对比场景数值接近；构建通过 |

---

## 6. 验证方案

- **构建**：每阶段增量构建（MSVC+nvcc Release）。
- **数值**：XPBD 结果与归档前旧实现（git 历史中可用）在相同场景/参数下对比
  （顶点轨迹或末帧几何）；旧实现有 `x_residual/time_spans` 记录可参考。
- **运行**：`main resource/scene/test_XPBD_ADMM.toml resource`（[solver] type=xpbd）。
- **注意**：XPBD 参数（stiffness 缩放、substeps）需与论文实验一致——以场景文件
  现有 `[xpbd]`/材质字段为准，必要时与用户核对。

---

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 归档前旧 XPBD 数值行为未知/不稳定 | 以 Macklin 公式 + Bender 求解循环为基准重写；旧代码仅作数学参考 |
| 刚度/参数与论文实验不匹配 | 场景参数逐项核对；提供 toml 覆盖字段 |
| 大场景 CPU 接触解算慢 | 接触约束类型分块并行（TBB，与 ADMM 一致）；后期 BVH_GPU 扩展 |
| 与 ADMM 显示/导出流程耦合 | 坚持只走 Solver 虚接口 + ICollisionDetector，main 零改动 |

---

---

## 实施进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| P1 约束复活 | ✅ 2026-03 | `src/constraint/xpbd/`（xpbd_constraint.h + 3 个实现 + 工厂 + mesh 构建）；数学复用共享 `xpbd_utils.h` |
| P2 求解器骨架 | ✅ | `XPBDSolver : Solver`；`XPBDConfig` + `[xpbd]` 段扩展；`SolverType::XPBD` + `create_solver` 返回 `Solver*`；app_initializer 分支装配；`AppContext::inner` 取代 main 的 cast；旧场景 `[xpbd] enable=true` 兼容选择 |
| P3 弹性约束 | ✅ | step 完整 XPBD 循环（substep → 重力/积分 → 迭代解弹性约束 → 速度阻尼 0.9995）；FEMTriangle/XPBD_FEMTet/Bending 接入 |
| P4 碰撞与摩擦 | ✅ | ProximalQuery 检测（dcd_interval）→ PT/EE 距离约束（thickness + contact_stiffness，5 遍 GS/迭代）；摩擦锥为后续扩展（mu 字段已预留） |
| P5 验证与收尾 | ✅（基本） | rod 场景 headless 冒烟：4 帧 1.3s、exit 0、碰撞计数正确；数值对比待用户视觉/实验确认；顺带修复 headless 模式仿真不启动的 bug |

**后续可选**：摩擦锥投影（mu）、BVH_GPU 接触、接触解算并行化（TBB，同 ADMM 分块思路）、
`use_GS_contact` 语义清理（当前独立实现恒用距离约束解算，字段仅兼容解析）。

*调研链接见 §1。*
