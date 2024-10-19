#pragma once

#include "mutils/common_types.h"
#include "compact_sparse_matrix.h"
#include <thrust/device_vector.h>
#include <cstdio>

namespace thrust {

    template <typename T1, typename T2>
    class pair;

    template <typename T>
    class device_allocator;

    template <typename T, typename Alloc>
    class device_vector;
}  // namespace thrust

struct CuCompactSparseMat {
    thrust::device_vector<ADU::Real> diag_data{};
    thrust::device_vector<ADU::Real> offdiag_data{}; // [Values]
    thrust::device_vector<int> offdiag_indices{}; // [inner indices]
    thrust::device_vector<int> offdiag_perline_start{}; // [), size = n+1, [outer Starts]

    int rows{};
    int cols{};

    explicit CuCompactSparseMat(const CompactSparseMat& hostData) ;

    CuCompactSparseMat(const CuCompactSparseMat& hostData) = delete;
    CuCompactSparseMat& operator=(const CuCompactSparseMat& hostData) = delete;
};


struct CuSolverData {

    CuCompactSparseMat sp_mat_device;
    thrust::device_vector<ADU::Real> x_curr_device{};
    thrust::device_vector<ADU::Real> b_curr_device{};

    thrust::device_vector<ADU::Real> jacobi_buffer_1;
    thrust::device_vector<ADU::Real> jacobi_buffer_2;

    explicit CuSolverData(const CompactSparseMat& hostData, int jacobi_buffer_size, int x_curr_size, int b_curr_size);

    CuSolverData(const CuSolverData& hostData) = delete;
    CuSolverData& operator=(const CuSolverData& hostData) = delete;
};

void copy_mat2thrustvector(const ADU::Matf_X3& mat, thrust::device_vector<ADU::Real>& tar, int rows) ;

void copy_thrustvector2mat(thrust::device_vector<ADU::Real>& vec, ADU::Matf_X3& tar , int rows) ;

void cu_jacobi_global(const CuCompactSparseMat& A,
    const thrust::device_vector<ADU::Real>& b,
    thrust::device_vector<ADU::Real>& x_curr,
    thrust::device_vector<ADU::Real>& jacobi_buffer_1,
    thrust::device_vector<ADU::Real>& jacobi_buffer_2,
    int nDynVert, int iter_cnt);

void convert_Mat2dvector(const ADU::Matf_X3& mat, thrust::device_vector<ADU::Vecf_3>& vec);