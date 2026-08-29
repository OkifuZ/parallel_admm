# 冗余 / 死代码清理记录（议题 19–30）

**日期**: 2025-02-21  
**关联清单**: `code-cleanup-and-optimization.md` 第五部分

---

## 变更概要

| 序号 | 问题 | 处理方式 |
|-----|------|----------|
| 19 | `int num = (app.bvh->bvs.size()) / 2` 未使用 | 删除（14–18 重构时已移除） |
| 20 | `int i = j` 多余 | 直接使用 `j`（14–18 重构时已修复） |
| 21 | `solver` 与 `solver_ptr` 重复声明、混淆 | 移除 `main_loop` 开头的未使用 `solver`，删除 step 块内未使用的 `solver_ptr` |
| 22 | 注释掉的 `//app.bvh->update()` | 删除 |
| 23 | 注释掉的 contact debug 输出块 | 删除 |
| 24 | `int n = Eigen::nbThreads()` 未使用 | 删除 |
| 25 | 大块 pin debug 注释 | 随 mesh 加载提取已移除 |
| 26 | 注释掉的 contact_w 赋值 | 随 solver 配置提取已移除 |
| 27 | 注释掉的 simpleBVH 备用实现 | 删除 |
| 28 | rod_twist 硬编码逻辑 | 此前已移除 |
| 29 | 全局 `enable_large_scale = false` 未使用 | 删除 |
| 30 | APP 成员 `vis_eect` 未使用 | 删除成员及对应注释掉的注册代码 |
