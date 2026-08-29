# main.cpp 结构优化记录（议题 14–18）

**日期**: 2025-02-21  
**关联清单**: `code-cleanup-and-optimization.md` 第四部分

---

## 变更概要

| 议题 | 问题 | 处理方式 |
|-----|------|----------|
| 14 | main.cpp 过长、职责过多 | 提取 mesh 加载、导出、solver 配置到独立模块 |
| 15 | 全局变量分散（app, app_config, out_path, step_idx） | 合并为 `AppContext g_ctx` |
| 16 | tri/tet/static 网格加载逻辑重复 | 提取 `load_meshes_from_config()` |
| 17 | OBJ/PLY 导出逻辑重复 | 提取 `export_frame()` |
| 18 | solver 配置逐行赋值冗长 | 提取 `apply_solver_config()` / `apply_parallel_solver_config()` |

---

## 新增文件

### 1. `src/mediator/mesh_from_config.h`
- **职责**: 从 APPConfig 加载所有网格，统一 tri/tet/static 流程
- **接口**: `load_meshes_from_config(mesh_data, config, resource_path) -> MeshLoadResult`
- **MeshLoadResult**: `mesh_id_list`, `material_list`, `static_mesh_id_begin`, `static_vert_begin`, `mesh_animate_info`

### 2. `src/mediator/export_frame.h`
- **职责**: 帧导出为 OBJ 或 PLY
- **接口**: `export_frame(out_path, scene_name, step_idx, verts, surface_tris, mesh_data, use_bin, separate_out)`
- **行为**: 支持 `separate_out` 分网格导出与单文件导出

### 3. `src/mediator/solver_config_applier.h`
- **职责**: 将 APPConfig 应用到 ADMM solver
- **接口**:
  - `apply_solver_config(solver, cfg, SolverSetupContext)`：CPU/GPU 通用配置
  - `apply_parallel_solver_config(parallel_solver, cfg)`：GPU 路径专属（Global_Jacobi_iter）
- **SolverSetupContext**: 包含 `mass`, `dt`, `is_static`, `static_mesh_id_begin`, `static_vert_begin`, `surf_vinds_set`, `mesh`

---

## main.cpp 修改

1. **全局状态**：`APP app; APPConfig app_config; path out_path; size_t step_idx` → `AppContext g_ctx`（含 app, config, out_path, step_idx）
2. **访问方式**：所有使用处改为 `g_ctx.app`, `g_ctx.config`, `g_ctx.out_path`, `g_ctx.step_idx`
3. **网格加载**：原 tri/tet/static 三段循环 → 单次 `load_meshes_from_config(...)`
4. **导出**：原 207–236 行分支逻辑 → 单行 `export_frame(...)`
5. **solver 配置**：原 365–405 行赋值 → `apply_solver_config()` + `apply_parallel_solver_config()`
6. **update_bvh_draw**：移除未用变量 `num`，`int i = j` 改为直接用 `j`

---

## 行数变化（约）

- main.cpp：约 520 行 → 约 525 行（include 增加，但逻辑更集中）
- 新增：mesh_from_config.h ~90 行，export_frame.h ~60 行，solver_config_applier.h ~75 行
