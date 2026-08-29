# 构建阻碍与修改记录

记录 `parallel_admm` 在 Windows 下首次构建时遇到的阻碍及解决方式。环境：vcpkg 路径 `C:\codebase\vcpkg`，CUDA 12.4，MSVC 2022。

---

## 1. CMAKE_CUDA_ARCHITECTURES 为空

**现象：**
```
CMake Error: CMAKE_CUDA_ARCHITECTURES must be non-empty if set.
```

**原因：** CMakeLists.txt 中设置了 `set(CMAKE_CUDA_ARCHITECTURES 86)`，但 vcpkg 工具链在配置阶段可能覆盖或清空该变量。

**解决：** 在 `cmake` 命令行显式传入：
```powershell
cmake .. -DCMAKE_TOOLCHAIN_FILE=... -DCMAKE_CUDA_ARCHITECTURES=86
```

---

## 2. polyscope 子模块未初始化

**现象：**
```
The source directory .../3rdparty/polyscope does not contain a CMakeLists.txt file.
```

**原因：** polyscope 为 git submodule，clone 后需单独初始化。

**解决：**
```powershell
git submodule update --init --recursive
```
需在非沙盒/完整权限下执行（否则可能遇到 Git/msys 的 signal pipe 错误）。

---

## 3. polyscope 依赖子模块（glfw、glm 等）缺失

**现象：**
```
The source directory .../3rdparty/polyscope/deps/glfw does not contain a CMakeLists.txt file.
...
target_compile_features no known features for CXX compiler "MSVC"
```

**原因：** polyscope 自身也有 submodule（glfw、glm、happly、imgui），仅 `--init` 不会递归拉取。

**解决：** 使用 `--init --recursive` 递归初始化所有嵌套子模块。

---

## 4. jacobi_solver.h 中 Thrust 前向声明冲突

**现象：** CUDA 编译 `mcuda_wrapper.cu` 时报错：
```
"thrust::device_vector" is ambiguous
identifier "vec" is undefined
expected a ")"
...
```

**原因：** `jacobi_solver.h` 中自定义了 thrust 命名空间的前向声明：
```cpp
namespace thrust {
    template <typename T>
    class device_vector;
    // ...
}
```
与 `#include <thrust/device_vector.h>` 提供的真实 Thrust 类型冲突，nvcc 解析时产生歧义。

**解决：** 删除 `jacobi_solver.h` 中的自定义 thrust 前向声明，只保留 `#include <thrust/device_vector.h>`。

---

## 5. export_frame.h 中 const 正确性

**现象：**
```
不能将"this"指针从"const MeshData"转换为"MeshData &"
MeshData::getFaceInds(int)
```

**原因：** `export_frame()` 接收 `const MeshData* mesh_data`，调用 `mesh_data->getFaceInds()` 时，`getFaceInds` 未声明为 const 成员函数。

**解决：** 将 `MeshData::getFaceInds(int mesh_id)` 声明为 `getFaceInds(int mesh_id) const`（该函数不修改对象状态）。

---

## 6. mesh_from_config.h 中 constexpr 与 UDL

**现象：**
```
表达式的计算结果不是常数
对未定义的函数或为未声明为"constexpr"的函数的调用导致了故障
ADU::operator ""_r 的用法
```

**原因：** `constexpr Real PI = 3.1415926_r` 要求字面量运算符 `operator "" _r` 在编译期可求值，原定义仅为 `inline`，非 `constexpr`。

**解决：** 在 `common_types.h` 中将 `inline Real operator "" _r(long double value)` 改为 `constexpr Real operator "" _r(long double value)`。

---

## 7. Thrust ABI 不匹配（C++ 与 CUDA 混编）

**现象：** 链接阶段大量 LNK2019：
```
无法解析的外部符号 op_Ax(..., thrust::THRUST_200301___CUDA_ARCH_LIST___NS::device_vector<...> &, ...)
已定义且可能匹配的符号: ... thrust::THRUST_200301_860_NS::device_vector<...> ...
```

**原因：** `admm_parallel_global_solver.cpp` 由 MSVC 编译，使用主机的 Thrust；jacobi_solver、compact_sparse_matrix_util 等 `.cu` 由 nvcc 编译，使用 CUDA 架构相关的 Thrust 命名空间（`THRUST_200301_860_NS`）。两者 `thrust::device_vector` 的 ABI 不同，符号无法匹配。

**解决：** 将 `admm_parallel_global_solver.cpp` 重命名为 `admm_parallel_global_solver.cu`，使其由 nvcc 编译，与其余 CUDA 代码统一使用同一套 Thrust ABI。CMake 中 `file(GLOB ... src/solver/*.cu)` 会自动包含该文件。

---

## 8. LNK4098 警告（可选处理）

**现象：**
```
warning LNK4098: 默认库"LIBCMT"与其他库的使用冲突；请使用 /NODEFAULTLIB:library
```

**说明：** 通常由 vcpkg/第三方库的运行时库（/MD vs /MT）与本工程不一致引起。当前构建可正常完成，可暂时忽略。若需消除，可在 CMakeLists 中统一设置 `/MD` 或按需添加 `/NODEFAULTLIB`。

---

## 完整构建命令

```powershell
# 1. 初始化子模块
cd D:\projects\parallel_admm
git submodule update --init --recursive

# 2. 配置
mkdir build -ErrorAction SilentlyContinue
cd build
$env:VCPKG_ROOT = "C:\codebase\vcpkg"
cmake .. -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DCMAKE_CUDA_ARCHITECTURES=86

# 3. 构建
cmake --build . --config Release
```

输出：`build/Release/main.exe`
