# 其它结构性问题清理记录（议题 38–42）

**日期**: 2025-02-21  
**关联清单**: `code-cleanup-and-optimization.md` 第八部分

---

## 变更概要

| 序号 | 问题 | 处理方式 |
|-----|------|----------|
| 38 | include 组织混乱 | 按「标准库 → 第三方 → CUDA/设备 → 项目」分组，各组内按字母序排列 |
| 39 | `#include"mutils/cnpy.h"` 缺少空格 | 改为 `#include "mutils/cnpy.h"` |
| 40 | `Eigen::setNbThreads(12)` 魔法数字 | 提取为常量 `kEigenNumThreads = 12` |
| 41 | `make_exception` 与 `throw std::runtime_error` 混用 | thickness_validation_check 失败处改为 `throw std::runtime_error`，与 export_frame 等保持一致 |
| 42 | `printf` 与 `std::cout` 混用 | main.cpp 中全部改为 `std::cout` |

---

## 38：include 组织

分组顺序：
1. **Standard library**: `<filesystem>`, `<fstream>`, `<functional>`, `<iostream>`, `<memory>`, `<string>`, `<unordered_set>`
2. **Third-party**: argparse, Eigen, igl, thrust, toml++; polyscope
3. **CUDA / device**: device_launch_parameters, lbvh, collision_detection_cuda, mutils cuda
4. **Project**: constraint, contact, mediator, mesh, mutils, solver, src_config

---

## 40：魔法数字说明

- `kEigenNumThreads`：仅抽取 main.cpp 中 `Eigen::setNbThreads(12)`
- `admm_max_iter`、`0.9995` 阻尼等位于 toml 配置或其它源文件，未改动
