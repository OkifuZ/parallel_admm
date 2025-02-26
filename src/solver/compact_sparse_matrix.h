#pragma once

#include "mutils/common_types.h"
#include "mutils/exception_handle.h"
#include "mutils/cformat.h"
#include <mutils/common_type_hostonly.h>

#include <vector>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/config.h>


struct CompactSparseMat{
    std::vector<ADU::Real> diag_data; // [Values]
    std::vector<ADU::Real> offdiag_data; // [Values]
    std::vector<int> offdiag_indices; // [inner indices]
    std::vector<int> offdiag_perline_start; // [), size = n+1, [outer Starts]

    std::vector<ADU::Real> data;
    std::vector<int> row_inds;
    std::vector<int> row_offsets;

    thrust::device_vector<ADU::Real> data_cu;
    thrust::device_vector<int> row_inds_cu;
    thrust::device_vector<int> row_offsets_cu;


    bool as_system_matrix{false};

    int rows{};
    int cols{};

    void buildSystemMatrix(const ADU::SpMatf &mat);

    void buildNormalMatrix(const ADU::SpMatf &mat);

    // equal to SparseMatrix::makeCompressed(), implement for possible manipulation
    explicit CompactSparseMat(const ADU::SpMatf& mat, bool asSystemMatirx=true) {
        this->as_system_matrix = asSystemMatirx;
        // mat is row major
        // each line has at least 1 value
        offdiag_perline_start.push_back(0);

        cols = mat.cols();
        rows = mat.rows();

        if (asSystemMatirx) {
            buildSystemMatrix(mat);
        }
        else {
            buildNormalMatrix(mat);
        }
    }
};

struct DataTemp {
    thrust::device_vector<ADU::Real> data;
    int cur_row = 0;
    int cur_col = 0;
    // TODO multithread


    static DataTemp& getInstance(int rows=-1, int cols=-1);




private:
    DataTemp() {}

};

void CUMat_Ax(const CompactSparseMat& A, const ADU::Real* x, ADU::Real* result);
void CUVec_a_plus_b(const ADU::Real* a, const ADU::Real* b, ADU::Real* result, int rows);
void CUVec_a_minus_b(const ADU::Real* a, const ADU::Real* b, ADU::Real* result, int rows);
void CUVec_scale(const ADU::Real* a, ADU::Real scale, ADU::Real* result, int rows);

// must be single thread
void op_Ax(const CompactSparseMat& A, thrust::device_vector<ADU::Real>& x, thrust::device_vector<ADU::Real>& result);
void op_a_plus_b(const thrust::device_vector<ADU::Real>& a, const thrust::device_vector<ADU::Real>& b, thrust::device_vector<ADU::Real>& result, int rows);
void op_a_minus_b(const thrust::device_vector<ADU::Real>& a, const thrust::device_vector<ADU::Real>& b, thrust::device_vector<ADU::Real>& result, int rows);
void op_scale(const thrust::device_vector<ADU::Real>& a, ADU::Real scale, thrust::device_vector<ADU::Real>& result, int rows);



//void resizeThrust(thrust::device_vector<ADU::Real>& data, int newSize, ADU::Real defaultVal = 0.0);
//void resizeThrust(thrust::device_vector<float4>& data, int newSize, float4 defaultVal);
//void resizeThrust(thrust::device_vector<float3>& data, int newSize, float3 defaultVal);

template <typename T>
void resizeThrust(thrust::device_vector<T>& data, int newSize, T defaultVal);

extern template void resizeThrust<float>(thrust::device_vector<float>&, int, float);
extern template void resizeThrust<double>(thrust::device_vector<double>&, int, double);
extern template void resizeThrust<float3>(thrust::device_vector<float3>&, int, float3);
extern template void resizeThrust<float4>(thrust::device_vector<float4>&, int, float4);
extern template void resizeThrust<int>(thrust::device_vector<int>&, int, int);

