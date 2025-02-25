#pragma once

#include "compact_sparse_matrix.h"


DataTemp& DataTemp::getInstance(int rows, int cols) {
    static DataTemp instance;
    if ((rows > 0 && cols > 0) && (instance.cur_row != rows || instance.cur_col != cols)) {
        if (instance.cur_row == -1) {
            instance.data.reserve(100000);
        }
        instance.cur_row = rows;
        instance.cur_col = cols;
        instance.data.resize(rows * cols, 0);
    }
    return instance;
}




__global__ void DoAxb(const ADU::Real* data, const int* row_inds, const int* row_offsets,
    const ADU::Real* b, ADU::Real* result,
    int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int start_id{};
    int end_id{};

    if (row < num_rows) {
        start_id = row_offsets[row];
        end_id = row_offsets[row+1];
        result[3 * row + 0] = 0;
        result[3 * row + 1] = 0;
        result[3 * row + 2] = 0;
        for (int i = start_id; i < end_id; i++) {
            int col = row_inds[i];
            ADU::Real val = data[i];
            result[3 * row + 0] += val * b[col * 3 + 0];
            result[3 * row + 1] += val * b[col * 3 + 1];
            result[3 * row + 2] += val * b[col * 3 + 2];
        }
    }
}

void op_Ax(const CompactSparseMat& A, thrust::device_vector<ADU::Real>& x, thrust::device_vector<ADU::Real>& result)
{
    CUMat_Ax(A, x.data().get(), result.data().get());
}

void CUMat_Ax(const CompactSparseMat& A, const ADU::Real* x, ADU::Real* result)
{
    if (x == nullptr) return;
    if (result == nullptr) return;
    const int rows = A.rows;
    const int cols = A.cols;

    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    auto& temp = DataTemp::getInstance(rows, 3);

    DoAxb<<<numBlocks, blockSize>>>(
        A.data_cu.data().get(), A.row_inds_cu.data().get(), A.row_offsets_cu.data().get(),
        x, temp.data.data().get(), rows);

    if (result != temp.data.data().get()) {
        cudaMemcpy(result, temp.data.data().get(), sizeof(ADU::Real) * rows * 3, cudaMemcpyDeviceToDevice);
    }
}

__global__ void Do_a_plus_b(const ADU::Real* a, const ADU::Real* b, int num_rows, ADU::Real* result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        result[row*3+0]= a[row*3 + 0] + b[row*3 + 0];
        result[row*3+1]= a[row*3 + 1] + b[row*3 + 1];
        result[row*3+2]= a[row*3 + 2] + b[row*3 + 2];
    }
}

void op_a_plus_b(const thrust::device_vector<ADU::Real>& a, const thrust::device_vector<ADU::Real>& b, thrust::device_vector<ADU::Real>& result, int rows)
{
    CUVec_a_plus_b(a.data().get(), b.data().get(), result.data().get(), rows);
}


void CUVec_a_plus_b(const ADU::Real* a, const ADU::Real* b, ADU::Real* result, int rows)
{
    if (a == nullptr || b == nullptr) return;

    int blockSize = 512;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    auto& temp = DataTemp::getInstance(rows, 3);

    Do_a_plus_b<<<numBlocks, blockSize>>>(a, b, rows, temp.data.data().get());

    if (result != temp.data.data().get()) {
        cudaMemcpy(result, temp.data.data().get(), sizeof(ADU::Real) * rows * 3, cudaMemcpyDeviceToDevice);
    }
}

__global__ void Do_a_minus_b(const ADU::Real* a, const ADU::Real* b, int num_rows, ADU::Real* result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        result[row*3+0]= a[row*3 + 0] - b[row*3 + 0];
        result[row*3+1]= a[row*3 + 1] - b[row*3 + 1];
        result[row*3+2]= a[row*3 + 2] - b[row*3 + 2];
    }
}


void op_a_minus_b(const thrust::device_vector<ADU::Real>& a, const thrust::device_vector<ADU::Real>& b, thrust::device_vector<ADU::Real>& result, int rows)
{
    CUVec_a_minus_b(a.data().get(), b.data().get(), result.data().get(), rows);
}


void CUVec_a_minus_b(const ADU::Real* a, const ADU::Real* b, ADU::Real* result, int rows)
{
    if (a == nullptr || b == nullptr) return;

    int blockSize = 512;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    auto& temp = DataTemp::getInstance(rows, 3);

    Do_a_minus_b<<<numBlocks, blockSize>>>(a, b, rows,  temp.data.data().get());

    if (result != temp.data.data().get()) {
        cudaMemcpy(result, temp.data.data().get(), sizeof(ADU::Real) * rows * 3, cudaMemcpyDeviceToDevice);
    }
}

__global__ void Do_scale(const ADU::Real* a, ADU::Real s, int num_rows, ADU::Real* result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        result[row*3+0]= a[row*3 + 0] * s;
        result[row*3+1]= a[row*3 + 1] * s;
        result[row*3+2]= a[row*3 + 2] * s;
    }
}

void op_scale(const thrust::device_vector<ADU::Real>& a, ADU::Real scale, thrust::device_vector<ADU::Real>& result, int rows)
{
    CUVec_scale(a.data().get(), scale, result.data().get(), rows);
}

void CUVec_scale(const ADU::Real* a, ADU::Real scale, ADU::Real* result, int rows) {
    if (a == nullptr || result == nullptr) return;

    int blockSize = 512;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    auto& temp = DataTemp::getInstance(rows, 3);

    Do_scale<<<numBlocks, blockSize>>>(a, scale, rows, temp.data.data().get());

    if (result != temp.data.data().get()) {
        cudaMemcpy(result, temp.data.data().get(), sizeof(ADU::Real) * rows * 3, cudaMemcpyDeviceToDevice);
    }
}

void resizeThrust(thrust::device_vector<ADU::Real>& data, int newSize, ADU::Real defaultVal)
{
    data.resize(newSize, defaultVal);
}

void CompactSparseMat::buildNormalMatrix(const ADU::SpMatf &mat) {
    row_offsets.push_back(0);
    for (int k = 0; k < mat.outerSize(); ++k) {

        int line_vals_num = 0;

        for (ADU::SpMatf::InnerIterator it(mat, k); it; ++it) {

            int col = it.col();
            int row = it.row();
            if (row != k) {
                ADU::make_exception(ADU::cformat("CompactSparseMat: each line has at least 1 value, empty in row %d", k));
            }

            line_vals_num += 1;
            data.push_back(it.value());
            row_inds.push_back(col);

        }
        row_offsets.push_back(row_offsets.back() + line_vals_num);
    }

    data_cu.insert(data_cu.end(), data.begin(), data.end());
    row_inds_cu.insert(row_inds_cu.end(), row_inds.begin(), row_inds.end());
    row_offsets_cu.insert(row_offsets_cu.end(), row_offsets.begin(), row_offsets.end());

}