cmake_minimum_required(VERSION 3.24)
# add_definitions( -DWIN32)
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else()
    message(FATAL_ERROR "VCPKG_ROOT environment variable is not set.")
endif()
set(CMAKE_CUDA_ARCHITECTURES 86)

SET(CMAKE_BUILD_TYPE Release)

project(admm_elasticity LANGUAGES C CXX CUDA)  

SET(CUDA_SEPARABLE_COMPILATION ON)
 
set(CMAKE_CXX_STANDARD 17) 
# set(CMAKE_CXX_FLAGS "/O2 /EHsc")  
set(CMAKE_CXX_FLAGS "/O2 /MP /EHsc")  

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>") 
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

set(CONFIG_IN_DIR "${PROJECT_SOURCE_DIR}/cmake.in")
configure_file(
"${CONFIG_IN_DIR}/src_config.h.in"
${PROJECT_BINARY_DIR}/config/src_config.h)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# set(ZLIB_ROOT "F:\\PBA\\ALIBS\\zlib_install")
set(ZLIB_USE_STATIC_LIBS ON)

# deps
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
# find_package(GMP REQUIRED)
# add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ccd")


# set(MKL_ARCH intel64)
# set(MKL_INTERFACE lp64)
# find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

file(GLOB
SOURCE_FILES
src/solver/*.cpp
src/constraint/*.cpp
src/mutils/*.cpp
src/contact/*.cpp
src/contact/simpleBVH/*.cpp
src/mesh/*.cpp
src/mediator/*.cpp

src/contact/lbvh/*.cu
src/constraint/*.cu
src/mutils/*.cu
src/solver/*.cu
src/contact/*.cu
# src/constraint/*.cu
# src/mutils/*.cu
# src/contact/*.cu
# src/mesh/*.cu
# src/mediator/*.cu
)



file(GLOB
HEADER_FILES
src/solver/*.h
src/constraint/*.h
src/mutils/*.h
src/contact/*.h
src/contact/simpleBVH/*.hpp
src/mesh/*.h
src/mediator/*.h

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

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${HEADER_FILES} ${SOURCE_FILES} 
)

include_directories(
    ${PROJECT_BINARY_DIR}/config src
    )

# add_executable(triangle
# ${HEADER_FILES} ${SOURCE_FILES} "src/triangle.cpp")
# target_link_libraries(triangle igl::core polyscope TBB::tbb Eigen3::Eigen mshio::mshio embree)

# add_executable(tetrahedral
# ${HEADER_FILES} ${SOURCE_FILES} "src/tetrahedral.cpp")
# target_link_libraries(tetrahedral polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree)
  
# add_executable(twobody
# ${HEADER_FILES} ${SOURCE_FILES} "src/twobody.cpp")
# target_link_libraries(twobody polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(main
${HEADER_FILES} ${SOURCE_FILES} "src/main.cpp"
        src/solver/mcuda_wrapper.h
        src/solver/mcuda_wrapper.cu
        src/mutils/common_type_hostonly.h
        src/constraint/triangle_constraint_device.h
        src/constraint/triangle_constraint_deivce.cu
        src/mutils/QR_reduced.h
        src/mutils/svd_fast3.cuh
        src/constraint/pin_constraint_deivce.cu
        src/constraint/pin_constraint_deivce.cu
        src/constraint/spring_constraint.h
        src/contact/contact_util.cu
        src/solver/contact_solver.cu
        src/solver/contact_solver.cuh
)
target_link_libraries(main polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree tomlplusplus::tomlplusplus argparse ZLIB::ZLIB)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# add_executable(testBVH
# ${HEADER_FILES} ${SOURCE_FILES} "src/testBVH.cpp")
# target_link_libraries(testBVH polyscope igl::core TBB::tbb Eigen3::Eigen mshio::mshio embree tomlplusplus::tomlplusplus argparse ZLIB::ZLIB)
# set_property(TARGET testBVH PROPERTY CUDA_SEPARABLE_COMPILATION ON)