# -----------------------------------------------------------------------------
# parallel_admm: ADMM soft body dynamics with frictional contact (CGI 2024)
# C++/CUDA, vcpkg for deps, polyscope for viz
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)
project(admm_elasticity LANGUAGES C CXX CUDA)

# --- Toolchain (vcpkg) ---
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else()
    message(FATAL_ERROR "VCPKG_ROOT environment variable is not set.")
endif()

# --- Build type & C++ ---
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "/O2 /MP /EHsc")

# --- CUDA ---
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_SEPARABLE_COMPILATION ON)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# --- Generated config ---
set(CONFIG_IN_DIR "${PROJECT_SOURCE_DIR}/cmake.in")
configure_file(
    "${CONFIG_IN_DIR}/src_config.h.in"
    "${PROJECT_BINARY_DIR}/config/src_config.h"
)

# --- CMake modules (deps) ---
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(ZLIB_USE_STATIC_LIBS ON)

include(eigen)
include(libigl)
igl_include(predicates)
include(mshio)
include(tomlplusplus)
include(argparse)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/polyscope")

find_package(TBB REQUIRED)
find_package(embree 4 REQUIRED)
find_package(ZLIB REQUIRED)

# --- Source tree (GLOB) ---
# src/solver, constraint, mutils, contact, mesh, mediator (cpp + cu where used)
file(GLOB SOURCE_FILES
    src/solver/*.cpp
    src/solver/core/*.cpp
    src/solver/backend_cpu/*.cpp
    src/constraint/*.cpp
    src/constraint/xpbd/*.cpp
    src/mutils/*.cpp
    src/contact/*.cpp
    src/contact/simpleBVH/*.cpp
    src/mesh/*.cpp
    src/app/*.cpp
    src/config/*.cpp
    src/contact/lbvh/*.cu
    src/constraint/*.cu
    src/mutils/*.cu
    src/solver/*.cu
    src/solver/backend_gpu/*.cu
    src/contact/*.cu
)

file(GLOB HEADER_FILES
    src/solver/*.h
    src/solver/core/*.h
    src/solver/backend_cpu/*.h
    src/solver/backend_gpu/*.h
    src/constraint/*.h
    src/constraint/xpbd/*.h
    src/mutils/*.h
    src/contact/*.h
    src/contact/simpleBVH/*.hpp
    src/mesh/*.h
    src/app/*.h
    src/config/*.h
    src/zensim/math/matrix/*.h
    src/zensim/math/matrix/*.hpp
    src/zensim/math/bit/*.h
    src/zensim/math/bit/*.hpp
    src/contact/lbvh/*.cuh
    src/mutils/*.cuh
    src/solver/*.cuh
    src/constraint/*.cuh
    src/contact/*.cuh
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES} ${SOURCE_FILES})

# --- Include dirs ---
include_directories(${PROJECT_BINARY_DIR}/config src ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# --- Main executable (scene from TOML, viz via polyscope) ---
add_executable(main
    ${HEADER_FILES}
    ${SOURCE_FILES}
    src/main.cpp
    src/solver/mcuda_wrapper.h
    src/solver/mcuda_wrapper.cu
    src/mutils/common_type_hostonly.h
    src/constraint/triangle_constraint_device.h
    src/constraint/triangle_constraint_device.cu
    src/mutils/QR_reduced.h
    src/mutils/svd_fast3.cuh
    src/constraint/pin_constraint_device.cu
    src/constraint/spring_constraint.h
    src/contact/contact_util.cu
    src/solver/contact_solver.cu
    src/solver/contact_solver.cuh
)
target_link_libraries(main
    polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree
    tomlplusplus::tomlplusplus argparse ZLIB::ZLIB
)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# --- Optional targets (uncomment to build) ---
# add_executable(triangle ${HEADER_FILES} ${SOURCE_FILES} src/triangle.cpp)
# target_link_libraries(triangle polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree)
# add_executable(tetrahedral ${HEADER_FILES} ${SOURCE_FILES} src/tetrahedral.cpp)
# target_link_libraries(tetrahedral polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree)
# add_executable(twobody ${HEADER_FILES} ${SOURCE_FILES} src/twobody.cpp)
# target_link_libraries(twobody polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree)
# add_executable(testBVH ${HEADER_FILES} ${SOURCE_FILES} src/testBVH.cpp)
# target_link_libraries(testBVH polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree tomlplusplus::tomlplusplus argparse ZLIB::ZLIB)
# set_property(TARGET testBVH PROPERTY CUDA_SEPARABLE_COMPILATION ON)
