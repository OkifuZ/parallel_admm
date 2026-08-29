# main.cpp 配置加载与 BVH 渲染封装记录

**日期**: 2025-02-22  
**关联**: 配置加载抽象、BVH 渲染封装、include 精简

---

## 变更概要

| 模块 | 问题 | 处理方式 |
|------|------|----------|
| 配置加载 | main 内配置/初始化逻辑过长 | 提取到 `app_initializer`：`parse_args()`、`init_app()` |
| BVH 渲染 | `update_bvh_draw()` 与 polyscope 逻辑混在 main | 提取到 `bvh_display_helper`：`update_bvh_geometry()` |
| 应用状态 | APP、AppContext 定义在 main.cpp | 提取到 `app_context.h` |
| include | main.cpp 含大量非直接依赖 | 精简至仅需的 20 行 |
| 构建 | 构建较慢 | build.bat 加 `--parallel` |
| 编译错误 | `make_exception` 未加命名空间 | `toml_to_config.h` 改为 `ADU::make_exception` |
| 链接 | `Mesh2Constraint::curr_start_row` 多重定义 | 改为 `inline static`，移除头文件外定义 |

---

## 新增文件

### 1. `src/mediator/app_context.h`
- **职责**: 定义应用状态 `APP` 和 `AppContext`
- **内容**: `APP`（solver、mesh、BVH、polyscope 句柄等）、`AppContext`（app、config、out_path、step_idx）
- **新增**: `APP::bvh_holder`（shared_ptr 持有 BVH），避免悬空指针

### 2. `src/mediator/app_initializer.h` / `app_initializer.cpp`
- **职责**: 命令行解析与完整应用初始化
- **接口**:
  - `parse_args(argc, argv) -> AppInitParams`：解析 config_path、resource_path
  - `init_app(ctx, params)`：加载 config、mesh、约束、solver、BVH、障碍物、animator
- **AppInitParams**: `config_path`, `resource_path`

### 3. `src/mediator/bvh_display_helper.h`
- **职责**: 将 BVH_GPU 的 AABB 转为 polyscope CurveNetwork 的 nodes/edges
- **接口**: `update_bvh_geometry(bvh, nodes, edges)`（inline 实现）
- **说明**: 原 `update_bvh_draw()` 逻辑移入此处

---

## 修改 / 删除文件

### `src/main.cpp`
- **行数**: 约 477 行 → 162 行
- **变更**:
  - 用 `parse_args()` / `init_app()` 替代原 200+ 行初始化
  - 用 `update_bvh_geometry()` 替代 `update_bvh_draw()`
  - include 从约 50 行减至 20 行（仅保留 filesystem、iostream、Eigen、polyscope、mediator、admm_solver、timer、src_config）

### `src/mediator/toml_to_config.h`
- **修复**: `make_exception(...)` → `ADU::make_exception(...)`（_CTML 模板内调用）

### `src/mediator/mesh_to_constraint.h`
- **修复**: `static size_t curr_start_row` + 头文件外定义 → `inline static size_t curr_start_row = 0`（消除 LNK2005）

### `build.bat`
- **优化**: `cmake --build . --config Release` → 加 `--parallel` 以并行编译

### 删除 `src/mediator/bvh_display_helper.cpp`
- **原因**: 实现合并到 `bvh_display_helper.h` 作为 inline 函数，避免链接问题

---

## main.cpp 精简后 include 列表

```cpp
#include <filesystem>
#include <iostream>
#include <Eigen/Core>
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "mediator/app_context.h"
#include "mediator/app_initializer.h"
#include "mediator/bvh_display_helper.h"
#include "mediator/contact_display_helper.h"
#include "mediator/export_frame.h"
#include "solver/core/admm_solver.h"
#include "mutils/timer.h"
#include "src_config.h"
```

---

## 验证

- 构建通过：`cmake --build . --config Release --parallel`
- main 逻辑依赖由 app_context、app_initializer、bvh_display_helper 等模块通过 include 链传入
