# 测试指南（parallel_admm）

分层测试流程：**构建 → 冒烟 → 数值 → 对比 → GUI 视觉**。每个 solver
（ADMM-CPU / ADMM-GPU / XPBD）都按此验证。

---

## 0. 构建

```bat
build.bat
```
或手动：
```bat
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --config Release --parallel
```
输出 `build/Release/main.exe`。冒烟测试前确保构建干净（改动大后建议
`--clean-first`，避免增量构建不一致，见历史教训）。

## 1. 冒烟测试（跑通 + 不崩 + 帧推进）

对每个 solver 跑一个小场景（如 `resource/scene/test_XPBD_ADMM.toml`，rod，
2520 顶点 / 6888 tets，秒级完成）：

```bat
:: ADMM (CPU)
build\Release\main.exe resource\scene\test_XPBD_ADMM.toml resource
:: XPBD（需要 [solver] type="xpbd" 或 [xpbd] enable=true 的临时场景）
```

**检查点**：
- `loading mesh... done loading mesh`、网格/约束打印正常
- `step frame 0/1/2...` 逐帧推进（headless 自动跑到 end_frame 退出）
- 退出码 0；无 `step_metrics.csv` 为空
- 输出目录出现 `<scene_name>\_00000.obj` 序列（export_obj=true 时）

**headless 冒烟模板**（临时场景副本）：
```toml
show_windows = false
export_obj = true
end_frame = 10
# [solver] type = "admm" 或 "xpbd"（或 [xpbd] enable = true 兼容旧文件）
```

## 2. 数值检查

- **无 NaN/Inf**：OBJ 顶点全为有限值（脚本/Excel 检查）。
- **能量/位移趋势**：自由下落场景位移单调增长、能量不爆炸；
  静止场景（g=0 + 全 pin）位移 ≈ 0（rod 场景两 solver 差异应 < 1e-3）。
- **对比脚本摘要**：mean/max vertex diff 数量级合理
  （同场景不同算法，典型 1e-3 ~ 1e-1，视场景而定）。

## 3. 动态对比（ADMM vs XPBD / CPU vs GPU）

```bash
python tools/compare_solvers.py \
    --scene resource/test_scene/armadillo_xpbd1.toml \
    --resource resource \
    --solvers admm,xpbd \
    --frames 5 \
    --exe build/Release/main.exe \
    --out build/compare_out
```

- 脚本自动：每个 solver 生成 headless 场景（solver 类型 / [xpbd] enable /
  export_obj / end_frame 自动打补丁）→ 跑 → 收集 OBJ + step_metrics.csv →
  `frames.csv`（逐帧 mean/max 顶点差、接触数、每帧耗时）+ `compare.png`
  （需 matplotlib：`pip install matplotlib`）。
- `--solvers admm_cpu,admm_gpu` 可对比 GPU 路径（GPU 场景需 `use_GPU=true`
  由脚本保持）。
- 大场景慢时降低 `--frames`；`--keep` 保留每个 solver 的原始输出目录复查。

## 4. GUI 视觉测试

```bat
build\Release\main.exe resource\scene\test_XPBD_ADMM.toml resource
```
- 窗口出现：网格可见；ImGui 面板有 pause/step/reset/export obj。
- 点 pause 取消暂停开始仿真；观察形变/接触点云（"contact" 图层默认隐藏，
  可勾选显示；接触点云尺寸与注册尺寸一致，否则 GUI 会报
  polyscope size validation 异常——遇此异常说明 points() 尺寸契约被破坏）。
- reset 按钮重置；export obj 勾选后在输出目录导出 OBJ 序列。

## 5. 回归清单（改动后必测）

| 项 | 命令/操作 | 预期 |
|----|-----------|------|
| ADMM-CPU headless | 冒烟模板 type=admm | exit 0，帧推进 |
| ADMM-GPU headless | 冒烟模板 type=admm + use_GPU=true | exit 0（需要 GPU） |
| XPBD headless | 冒烟模板 type=xpbd | exit 0，帧推进 |
| GUI 模式 | show_windows=true 跑几帧后关窗口 | 无 polyscope 异常 |
| 对比脚本 | compare_solvers.py 双跑 | frames.csv 非空 |
| 指标 CSV | step_metrics.csv | 每帧一行（frame,step_ms,n_contacts） |

## 6. 已知限制

- XPBD 接触解算为单线程（每迭代 5 遍 PT/EE），大场景（armadillo 2.8 万顶点）
  每帧可达数十秒；对比测试用 `--frames` 小值。
- ADMM-GPU 未在本机做完整运行时验证（需要 GPU 会话）。
- `ENABLE_ASAN` CMake 选项仅对 CXX 源码有效，与 CUDA 对象混编无法链接
  （已知限制，勿用于带 CUDA 的构建）。
