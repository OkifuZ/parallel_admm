# 代码规整与优化清单

基于对 `parallel_admm` 代码库的扫描整理，用于后续逐项处理。仅列出可改进点，不做修改记录。

---

## 一、拼写错误与命名

| 序号 | 位置 | 错误 | 建议修正 | 状态 |
|-----|------|------|---------|------|
| 1 | `mesh_container.h` | `roatate` | `rotate` | ✅ 已修改 |
| 2 | `mesh_container.h` | `centerlize` | `centralize` | ✅ 已修改 |
| 3 | `mesh_container.h` | `compuite_AABB` | `compute_AABB` | ✅ 已修改 |
| 4 | `mesh_container.h` | `inidices`（异常信息） | `indices` | ✅ 已修改 |
| 5 | `triangle_constraint_deivce.cu`, `pin_constraint_deivce.cu` | 文件名 `deivce` | `device` | ✅ 已修改 |
| 6 | `narrow_phase.cpp` | `narrow_pahse`（Timer 名） | `narrow_phase` | ✅ 已修改 |
| 7 | `toml_to_config.h` + 各 TOML + Python | `seperate_out` | `separate_out` | ✅ 已修改 |
| 8 | `mesh_phy_material.h` | `PhyxMaterial` | 可视作 Physics 缩写保留，或统一为 `PhysicsMaterial` | 待定 |

---

## 二、重复的 `#pragma once` / include

| 序号 | 位置 | 问题 | 状态 |
|-----|------|------|------|
| 9 | `admm_full_solver.h` | 文件头重复 `#pragma once` | ✅ 已修改 |
| 10 | `main.cpp` | 重复 `#include "Eigen/Core"`（约 40–42 行） | ✅ 已修改 |

---

## 三、逻辑 / 配置错误

| 序号 | 位置 | 问题 | 状态 |
|-----|------|------|------|
| 11 | `main.cpp` | BVH 策略逻辑错误 + rod_twist 硬编码 | ✅ 已修改 |
| 12 | `main.cpp` | `view::bgColor` 三通道都用了 `background_color.x()` | ✅ 已修改 |
| 13 | `admm_full_solver_RL_damping.cpp` | `step_cnt == 15` 时 admm_max_iter/残差打印等硬编码 | ✅ 已修改 |

---

## 四、main.cpp 结构与职责过重

| 序号 | 问题 | 状态 |
|-----|------|------|
| 14 | **长度**：main.cpp 约 520 行，承担入口、配置、初始化、主循环、可视化等 | ✅ 已修改 |
| 15 | **全局状态**：`APP app`、`APPConfig app_config`、`out_path`、`step_idx` 等全局变量 | ✅ 已修改 |
| 16 | **网格加载重复**：tri/tet/static 三段逻辑高度相似（add_mesh → center → scale → rotate → translate → material），可提取为 `load_single_mesh(mesh, resource_path)` 或类似函数 | ✅ 已修改 |
| 17 | **导出逻辑重复**：207–240 行 separate_out 分支下，OBJ/PLY 的写法和错误处理几乎相同，可抽成 `export_frame(...)` | ✅ 已修改 |
| 18 | **solver 配置**：365–394 行大量逐行赋值，可增加 `apply_to_solver(...)` 或类似集中接口 | ✅ 已修改 |

详见 `agent_aux/refactor-main-14-18.md`

---

## 五、冗余 / 死代码

| 序号 | 位置 | 说明 | 状态 |
|-----|------|------|------|
| 19 | `main.cpp` 130 | `int num = (app.bvh->bvs.size()) / 2` 定义后未使用 | ✅ 已修改 |
| 20 | `main.cpp` 135 | `int i = j` 可直接用 `j` | ✅ 已修改 |
| 21 | `main.cpp` 173 | `solver` 声明后，201 行又用 `solver_ptr`，命名与用途混淆 | ✅ 已修改 |
| 22 | `main.cpp` 204 | 注释掉的 `//app.bvh->update()` | ✅ 已修改 |
| 23 | `main.cpp` 273–278 | 大段注释掉的 contact debug 输出 | ✅ 已修改 |
| 24 | `main.cpp` 322 | `int n = Eigen::nbThreads()` 未使用 | ✅ 已修改 |
| 25 | `main.cpp` 386–390 | 大块注释（for pin debug） | ✅ 已修改 |
| 26 | `main.cpp` 457–458 | 注释掉的 contact_w 赋值 | ✅ 已修改 |
| 27 | `main.cpp` 567–569 | 注释掉的 simpleBVH 备用实现 | ✅ 已修改 |
| 28 | `main.cpp` | 固定的 rod_twist 场景特殊逻辑 | ✅ 已移除 |
| 29 | `main.cpp` 65 | 全局 `enable_large_scale = false` 未被使用 | ✅ 已修改 |
| 30 | `main.cpp` 97 | `vis_eect` 未使用 | ✅ 已修改 |

---

## 六、CPU / GPU 路径分支分散

| 序号 | 问题 |
|-----|------|
| 31 | 碰撞检测：CPU 用 ProximalQuery，GPU 用 BVH_GPU，接口不统一，main_loop 中需分支访问 contact 数据 |
| 32 | 接触可视化：261–268 行按 `!use_GPU` 分支，可封装为 `get_contact_points()` / `get_contact_normals()` 的虚函数或统一接口 |
| 33 | ADMM step 的 CPU / GPU 实现逻辑相近，但分散在不同文件，可考虑 Strategy 或模板抽象，减少重复 |

---

## 七、TODO / 未实现

| 序号 | 位置 | 内容 |
|-----|------|------|
| 34 | `admm_full_solver_RL_damping.cpp` 285 | `find_contact_islands()` 仅 TODO，未实现 |
| 35 | `admm_full_solver.h` 183 | `contact_islands` 成员定义但未使用 |
| 36 | `collision_detection_cuda.cpp` 121, 143 | “init got memory leak!” 需确认并处理 |
| 37 | `admm_parallel_global_solver.cpp` 384 | “will this actually work?” 需验证逻辑 |

---

## 八、其它结构性问题

| 序号 | 问题 | 状态 |
|-----|------|------|
| 38 | **include 组织**：main.cpp 顶部混排 STL、第三方、项目头文件，建议按：标准库 → 第三方 → 项目，并保持字母序 | ✅ 已修改 |
| 39 | **include 间距**：如 `#include"mutils/cnpy.h"` 缺少空格 | ✅ 已修改 |
| 40 | **魔法数字**：如 `Eigen::setNbThreads(12)` 等，建议抽成常量或配置 | ✅ 已修改 |
| 41 | **异常处理风格**：部分用 `make_exception`，部分用 `throw std::runtime_error`，建议统一 | ✅ 已修改 |
| 42 | **输出格式**：`printf` 与 `std::cout` 混用，建议统一用 `std::cout` 或日志库 | ✅ 已修改 |

详见 `agent_aux/cleanup-misc-38-42.md`

---

## 九、建议处理顺序

**高优先级（易改、影响大）：**  
- 1–7：拼写修正 ✅ 已完成  
- 9–10：去掉重复 include ✅ 已完成  
- 11–13：BVH 策略、背景色、移除硬编码与调试逻辑 ✅ 已完成    

**中优先级（结构优化）：**  
- 14–18：main.cpp 拆分与职责划分 ✅ 已完成  
- 19–30：删除未使用变量和死代码 ✅ 已完成  
- 38–42：include 组织、常量、异常/输出风格统一 ✅ 已完成  

**低优先级（需要设计决策）：**  
- 31–33：CPU/GPU 接口抽象  
- 34–37：TODO 实现与内存泄漏排查    

---

*文档生成自代码库扫描，后续可按序号逐项处理并在此文档中勾选或补充备注。*
