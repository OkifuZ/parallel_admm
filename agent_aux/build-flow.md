# 构建流程

Windows 下 `parallel_admm` 的标准构建流程。

## 前置条件

- **vcpkg**：已安装并设置 `VCPKG_ROOT`（如 `C:\codebase\vcpkg`）
- **CUDA**：12.x，与 CMake 能正确检测
- **MSVC**：Visual Studio 2022 或兼容版本
- **Git**：用于子模块

## 构建步骤

### 1. 初始化子模块

首次 clone 或子模块有更新时执行：

```batch
git submodule update --init --recursive
```

会拉取 polyscope 及其嵌套依赖（glfw、glm、happly、imgui 等）。

### 2. 配置 CMake

```batch
mkdir build 2>nul
cd build
set VCPKG_ROOT=C:\codebase\vcpkg
cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake -DCMAKE_CUDA_ARCHITECTURES=86
```

> 若已设置环境变量 `VCPKG_ROOT`，可省略 `set VCPKG_ROOT=...`。
> `CMAKE_CUDA_ARCHITECTURES=86` 对应 SM 8.6（如 RTX 30 系列），按 GPU 调整。

### 3. 编译

```batch
cmake --build . --config Release
```

### 4. 输出

- 可执行文件：`build\Release\main.exe`

## 一键脚本

根目录 `build.bat` 会依次执行上述步骤，用法：

```batch
build.bat
```

可通过环境变量 `VCPKG_ROOT` 指定 vcpkg 路径，默认 `C:\codebase\vcpkg`。

## 常见问题

见 [build-obstacles-and-fixes.md](build-obstacles-and-fixes.md)。
