# 模块化优化设计（Constraint / Contact / Mediator / Utils）

遵循 [refactor-guidelines.md](refactor-guidelines.md)。Solver 内部已完成 backend 抽象（IADMMBackend, IContactSolver, backend_cpu / backend_gpu），本文档聚焦 constraint、contact、mutils 的职责边界与优化方向。

**原则：mediator 不应作为独立模块存在**，其内容按职责归入 config、constraint、solver、contact、mesh、app 等模块。

**Nodal contact / Obstacle 归属**：保留在 contact 模块；constraint 通过 forward decl 或 contact 的窄接口访问，不 include contact 实现细节。

---

## 1. 当前依赖关系概览（mediator 拆散前）

```
main.cpp
  └── mediator (app_context, app_initializer, bvh_display_helper, contact_display_helper, export_frame)
        ├── solver, mesh, contact, constraint, mutils

solver/     └─ constraint, contact, mutils
contact/    └─ contact.h (Obstacle, ObstacleContactInfo) ← 保留
constraint/ └─ nodal_collision_constraint.h → contact/contact.h

mediator/   ← 拆散后删除
  ├── toml_to_config, mesh_from_config
  ├── mesh_to_constraint
  ├── solver_config_applier
  ├── DCD_validation_check
  ├── bvh_display_helper, contact_display_helper
  ├── export_frame
  └── app_context, app_initializer
```

---

## 2. Mediator 拆散：归属表

| 原 mediator 文件 | 迁入模块 | 新路径 |
|------------------|----------|--------|
| toml_to_config.h | config/ | config/app_config.h |
| mesh_from_config.h | config/ | config/mesh_from_config.h |
| mesh_to_constraint.h | constraint/ | constraint/mesh_to_constraint.h |
| solver_config_applier.h/.cpp | solver/ | solver/solver_config_applier.h, .cpp |
| DCD_validation_check.h | contact/ | contact/dcd_validation.h |
| bvh_display_helper.h | contact/ | contact/bvh_display.h |
| contact_display_helper.h/.cpp | contact/ | contact/contact_display.h, .cpp |
| export_frame.h | mesh/ | mesh/export_frame.h |
| app_context.h | app/ | app/app_context.h |
| app_initializer.h/.cpp | app/ | app/app_initializer.h, .cpp |
| toml_to_scene.h | — | 删除 |

拆散后顶层结构：

```
main.cpp └── app/
config/      ← toml 解析、mesh 加载（仅 mesh + mutils）
app/         ← 状态与编排
constraint/  ← 约束 + mesh_to_constraint
solver/      ← 求解器 + solver_config_applier
contact/     ← 碰撞/接触 + nodal/obstacle + 校验 + 显示
mesh/        ← 网格 + export_frame
mutils/      ← 基础类型与工具
```

---

## 3. 可执行 Phase 引导

每个 Phase 结束时：构建通过，运行 1–2 个 scene 回归。

---

### Phase 1：Config 模块

**目标**：建立 config/，与 solver/constraint/contact 解耦。

**步骤**：
1. 新建 `src/config/`。
2. `mediator/toml_to_config.h` → `config/app_config.h`（重命名 APPConfig、load_config 等，保留逻辑）。
3. `mediator/mesh_from_config.h` → `config/mesh_from_config.h`；依赖仅限 config、mesh、mutils。
4. 更新引用 `toml_to_config`、`mesh_from_config` 的 include 路径。
5. 验证：编译通过；若有独立测试/入口可跑 mesh 加载，执行一次。

**产出**：`config/app_config.h`、`config/mesh_from_config.h`；mediator 内对应文件可暂保留作对照，Phase 4 再删。

---

### Phase 2：Contact / Mesh 辅助迁移

**目标**：将 DCD 校验、BVH/接触显示、帧导出迁入 contact 与 mesh。

**步骤**：
1. `mediator/DCD_validation_check.h` → `contact/dcd_validation.h`。
2. `mediator/bvh_display_helper.h` → `contact/bvh_display.h`。
3. `mediator/contact_display_helper.h`、`.cpp` → `contact/contact_display.h`、`.cpp`。
4. `mediator/export_frame.h` → `mesh/export_frame.h`；去除对 toml_to_config 的 include，仅用 mesh + mutils/cformat + igl。
5. 更新所有引用上述文件的 include 路径。
6. 验证：构建 + 1–2 scene。

**产出**：contact 与 mesh 各自获得显示与导出相关文件；mediator 仍保留 toml、mesh_from、mesh_to_constraint、solver_config、app_context、app_initializer。

---

### Phase 3：Constraint / Solver 迁移

**目标**：迁移 mesh_to_constraint、solver_config_applier。

**步骤**：
1. `mediator/mesh_to_constraint.h` → `constraint/mesh_to_constraint.h`。
2. `mediator/solver_config_applier.h`、`.cpp` → `solver/solver_config_applier.h`、`.cpp`。
3. 更新 include 路径，确保 solver_config_applier 依赖 config 而非 mediator。
4. 验证：构建 + 1–2 scene。

**产出**：constraint 含 mesh_to_constraint；solver 含 solver_config_applier；mediator 仅剩 app_context、app_initializer。

---

### Phase 4：App 模块与 main 重写

**目标**：建立 app/，改写 main.cpp，删除 mediator。

**步骤**：
1. 新建 `src/app/`。
2. `mediator/app_context.h` → `app/app_context.h`。
3. `mediator/app_initializer.h`、`.cpp` → `app/app_initializer.h`、`.cpp`。
4. 重写 main.cpp：`#include "app/app_context.h"`、`"app/app_initializer.h"`；移除对 mediator 的引用。
5. 更新 CMakeLists.txt：mediator 从 GLOB 移除，app、config 加入（若 GLOB 未覆盖则显式添加）。
6. 删除 `src/mediator/` 目录。
7. 验证：构建 + 全流程 1–2 scene。

**产出**：无 mediator；main → app → config / solver / mesh / contact / constraint。

---

### Phase 5：Contact ↔ Constraint 解耦（Obstacle 保留 contact）

**目标**：constraint 不直接依赖 contact 内部实现；nodal contact / Obstacle 仍在 contact。

**步骤**：
1. 在 contact 中明确 Obstacle、ObstacleContactInfo 的 public 接口；必要时抽出 `contact/obstacle.h` 仅含类型定义。
2. 在 `nodal_collision_constraint.h` 中：用 `class Obstacle;` 等 forward decl；将需要完整定义的逻辑移到 .cpp 或在 constraint 内单独 .cpp include `contact/obstacle.h`。
3. 确保 constraint 不 include 除 obstacle 类型外的 contact 实现（如 narrow_phase、lbvh 等）。
4. 验证：构建 + 1–2 scene（含 obstacle 场景）。

**产出**：constraint 依赖 contact 的 Obstacle 接口，不依赖 contact 检测逻辑。

---

### Phase 6（可选）：进一步优化

**低优先级**，可延后：

| 项 | 说明 |
|----|------|
| Constraint 注册/Factory | mesh_to_constraint 注册 typeID→创建函数，solver 不直接 include 各 `*_constraint.h` |
| Contact ICollisionDetector | 定义接口，narrow_phase / collision_detection_cuda 实现，solver 通过接口调用 |
| mutils 子目录 | gpu/、math/ 分组，便于非 CUDA 构建与维护 |

---

## 4. 与 Solver 设计的衔接

Solver 已有 IADMMBackend、IContactSolver；constraint、contact 建议：

- **Constraint**：`IConstraint` 基类 + device kernel（`*_device.cuh`）；backend 负责调用。
- **Contact**：Obstacle 保留在 contact；constraint 通过 forward decl / 窄接口使用。
- **Sync 边界**：Host↔Device sync 仅在 init/reset/step；在 agent_aux 标注 [SYNC] 点。

---

## 5. 检查清单（对照 refactor-guidelines）

- [ ] 每个模块 public 接口清晰
- [ ] 头文件尽量 forward declare，实现放 .cpp/.cu
- [ ] 修改头文件时考虑 .cu 重编范围
- [ ] 每 Phase 后运行 1–2 scene 回归
- [ ] agent_aux 中记录 Host↔Device sync 点与数据流
