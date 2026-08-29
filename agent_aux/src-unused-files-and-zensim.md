# main 入口下的 src 冗余文件与 zensim 依赖

从 `main.cpp` 追溯依赖链，分析哪些文件对 main 可执行文件不必要，以及 zensim 的使用位置。

---

## 一、对 main 可执行文件无用的文件

### 1. 未被 main 目标编译的源文件（仅用于可选 target）

CMake GLOB 只匹配 `src/solver/`, `src/constraint/`, `src/contact/`, `src/mesh/`, `src/mediator/`, `src/mutils/` 下的 `.cpp`，`src/` 根目录的 `.cpp` 不参与 main 构建：

| 文件 | 说明 |
|------|------|
| `src/twobody.cpp` | 独立程序入口，使用 Mesh2Constraint，对应被注释的 `add_executable(twobody ...)` |
| `src/triangle.cpp` | 对应被注释的 triangle target |
| `src/tetrahedral.cpp` | 对应被注释的 tetrahedral target |
| `src/testBVH.cpp` | 对应被注释的 testBVH target |

### 2. 未被 main 依赖链引用的文件（推测可删）

| 文件 | 说明 |
|------|------|
| `src/mediator/toml_to_scene.h` | 空头文件，无 include |
| `src/mutils/helper_cuda.h` | 无其他文件 include |
| `src/mutils/helper_string.h` | 仅被 helper_cuda.h 使用，helper_cuda 未被使用 |
| `src/constraint/spring_constraint.h` | `Mesh2Constraint` 用 PinConstraint/TriangleConstraint 等，未用 SpringConstraint；仅 twobody 可能用 |

### 3. 仅被可选 target 使用的约束实现

| 文件 | 说明 |
|------|------|
| `src/constraint/XPBD_traingle.cpp` | XPBD 三角形，mesh_to_constraint 中相关代码被注释 |
| `src/constraint/XPBD_tetrahedron.cpp` | XPBD 四面体，mesh_to_constraint 中相关代码被注释 |
| `src/constraint/XPBD_bending.cpp` | XPBD 弯曲，mesh_to_constraint 中相关代码被注释 |

这些仍通过 GLOB 参与 main 构建（提供 XPBD_constraints.h 中的虚函数实现），删除前需确认 XPBD 模式是否在 main 中使用。

---

## 二、zensim 依赖位置

### 使用点

| 文件 | 用途 |
|------|------|
| `src/mutils/gpu_eigen_libs.cu` | `SVD()` 中调用 `zs::math::qr_svd()`，来自 `zensim/math/matrix/QRSVD.hpp`；并 include `zensim/math/bit/Bits.h` |
| `src/constraint/spring_constraint.h` | 使用 `zensim/math/VecInterface.hpp`（SpringConstraint 对 main 未使用） |

### 依赖链

```
main
  → contact/lbvh/lbvh.cu
       → mutils/gpu_eigen_libs.cuh
  → mutils/gpu_eigen_libs.cu (实现，含 zensim)
       → zensim/math/matrix/QRSVD.hpp
       → zensim/math/bit/Bits.h
```

lbvh 用于 GPU BVH 碰撞检测；tet/triangle 等 GPU 约束的 SVD 通过 `gpu_eigen_libs.cu` 的 `__GEIGEN__::SVD()` 调用 zensim 的 QR-SVD。

### 精简 zensim 依赖的做法

- **main 必需**：`gpu_eigen_libs.cu` 中的 SVD 需要 `QRSVD.hpp` 与 `Bits.h`
- **可考虑移除**：若删除 `spring_constraint.h` 或不再使用 SpringConstraint，可去掉 `VecInterface.hpp` 依赖

---

## 三、小结

| 类别 | 文件 |
|------|------|
| main 不编译（仅 optional target） | twobody.cpp, triangle.cpp, tetrahedral.cpp, testBVH.cpp |
| 未被 main 依赖 | toml_to_scene.h, helper_cuda.h, helper_string.h, spring_constraint.h |
| zensim 必需 | gpu_eigen_libs.cu → QRSVD.hpp, Bits.h |
| zensim 可选 | spring_constraint.h → VecInterface.hpp（若移除 SpringConstraint） |
