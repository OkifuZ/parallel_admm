#include "solver/jacobi_solver.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <cuda_runtime.h>
#include "mutils/cuda_tools.cuh"
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

using namespace ADU;


CuCompactSparseMat::CuCompactSparseMat(const CompactSparseMat& hostData) {
    rows = hostData.rows;
    cols = hostData.cols;
    diag_data = hostData.diag_data;
    offdiag_data = hostData.offdiag_data;
    offdiag_indices = hostData.offdiag_indices;
    offdiag_perline_start = hostData.offdiag_perline_start;
}

CuSolverData::CuSolverData(const CompactSparseMat& hostData,
    int jacobi_buffer_size, int x_curr_size, int b_curr_size, int p_size, int constraint_dim) :
    sp_mat_device(hostData)
{
    jacobi_buffer_1.resize(jacobi_buffer_size * 3, 0);
    jacobi_buffer_2.resize(jacobi_buffer_size * 3, 0);
    x_curr_device.resize(x_curr_size * 3, 0);
    x_0_device.resize(x_curr_size * 3, 0);
    v_0_device.resize(x_curr_size * 3, 0);
    b_curr_device.resize(b_curr_size * 3, 0);
    z_buffer.resize(constraint_dim * 3, 0);
    p_device.resize(p_size * 3, 0);
}



void convert_Mat2dvector(const ADU::Matf_X3& mat, thrust::device_vector<ADU::Vecf_3>& vec) {
    std::vector<ADU::Vecf_3> hostVector(mat.rows());
    for (int i = 0; i < mat.rows(); ++i) {
        hostVector[i] = ADU::Vecf_3(mat.row(i));
    }
    if (!vec.empty()) {
        vec.clear();
    }
    vec.insert(vec.end(), hostVector.begin(), hostVector.end());
}


struct JacobiFunctor {
    const ADU::Real* diag_data;
    const ADU::Real* offdiag_data;
    const int* offdiag_indices;
    const int* offdiag_perline_start;
    const Real* b; // Pointer to b matrix data on the device
    const Real* x_curr; // Pointer to x_curr matrix data on the device
    Real* jacobi_buffer; // Pointer to jacobi_buffer data on the device

    int rows; // Number of rows (nDynVert)

    JacobiFunctor(
        const ADU::Real* diag_data, const ADU::Real* offdiag_data,
        const int* offdiag_indices, const int* offdiag_perline_start,
        const Real* b, const Real* x_curr, Real* jacobi_buffer,
        int rows)
        : diag_data(diag_data), offdiag_data(offdiag_data),
          offdiag_indices(offdiag_indices), offdiag_perline_start(offdiag_perline_start),
          b(b), x_curr(x_curr), jacobi_buffer(jacobi_buffer),
          rows(rows) {}

    __device__ void operator()(int i) const {

        Real a_ii = diag_data[i];

        // Load b_i_p (Vecf_3 is assumed to be an array of size 3)
        Real b_i_p[3] = {b[i * 3], b[i * 3 + 1], b[i * 3 + 2]};

        Real b_i_m[3] = {0.0, 0.0, 0.0};
        int st_ind = offdiag_perline_start[i];
        int ed_ind = offdiag_perline_start[i + 1];
        for (int j = st_ind; j < ed_ind; j++) {
            int ind = offdiag_indices[j]; // col

            for (int k = 0; k < 3; ++k) {
                b_i_m[k] += x_curr[ind * 3 + k] * offdiag_data[j];
            }
        }

        // Compute (b_i_p - b_i_m) / a_ii and store in jacobi_buffer
        for (int k = 0; k < 3; ++k) {
            jacobi_buffer[i * 3 + k] = (b_i_p[k] - b_i_m[k]) / a_ii;
        }
    }
};

void copy_mat2thrustvector(const ADU::Matf_X3& mat, thrust::device_vector<ADU::Real>& tar, int len) {
    if (mat.rows() < len || tar.size() < len * 3) {
        throw std::runtime_error("Size of vec must be 3 times the number of rows in tar.");
    }
    thrust::copy(mat.data(), mat.data() + len * 3, tar.data());
}

void copy_vec2thrustvector(const ADU::Vecf_X& vec, thrust::device_vector<ADU::Real>& tar, int len) {
    if (vec.rows() < len || tar.size() < len) {
        throw std::runtime_error("Size of vec must be 3 times the number of rows in tar.");
    }
    thrust::copy(vec.data(), vec.data() + len, tar.data());
}


void copy_thrustvector2mat(thrust::device_vector<ADU::Real>& vec, ADU::Matf_X3& tar, int len) {
    // Check if the size of vec matches 3 times the number of rows in tar
    if (tar.rows() < len || vec.size() < len * 3) {
        throw std::runtime_error("Size of vec must be 3 times the number of rows in tar.");
    }

    thrust::copy(vec.begin(), vec.begin() + len * 3, tar.data());
}

void copy_thrustvector2thrust(thrust::device_vector<ADU::Real>& vec, thrust::device_vector<ADU::Real>& tar, int size) {
    // Check if the size of vec matches 3 times the number of rows in tar
   
    thrust::copy(vec.begin(), vec.begin() + size, tar.data());
}


void cu_jacobi_global(const CuCompactSparseMat& A,
    const thrust::device_vector<ADU::Real>& b,
    thrust::device_vector<Real>& x_curr,
    thrust::device_vector<Real>& jacobi_buffer_1,
    thrust::device_vector<Real>& jacobi_buffer_2,
    int nDynVert, int iter_cnt)
{

    Real* current_buffer = thrust::raw_pointer_cast(jacobi_buffer_1.data());
    Real* next_buffer = thrust::raw_pointer_cast(jacobi_buffer_2.data());

    JacobiFunctor functor(
        thrust::raw_pointer_cast(A.diag_data.data()),
        thrust::raw_pointer_cast(A.offdiag_data.data()),
        thrust::raw_pointer_cast(A.offdiag_indices.data()),
        thrust::raw_pointer_cast(A.offdiag_perline_start.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(x_curr.data()),
        current_buffer,
        nDynVert);

    for (int i = 0; i < iter_cnt; i++) {
        // read from x_curr
        // write to cur_buffer
        // set x_curr = cur_buffer
        // swap cur_buffer, nxt_buffer
        thrust::for_each(
           thrust::device,
           thrust::counting_iterator<int>(0),
           thrust::counting_iterator<int>(nDynVert),
           functor);

        thrust::swap(current_buffer, next_buffer);

        // Update functor to write to the next buffer
        functor.x_curr = next_buffer;
        functor.jacobi_buffer = current_buffer;
    }
    thrust::copy(functor.x_curr, functor.x_curr + nDynVert * 3, x_curr.begin());
}


__global__ void do_integration_kernel(int nVert, int nDynVert, const ADU::Real dt, const ADU::Real g, const ADU::Real* x_0, const int* is_fix, const ADU::Real* M,
    ADU::Real* v_0, ADU::Real* x_curr, ADU::Real* M_x_tilde)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nVert) {
        if (is_fix[idx] == 0) {
            if (idx < nDynVert) {
                v_0[idx * 3 + 1] -= g * dt;
                x_curr[idx * 3 + 0] = x_0[idx * 3 + 0] + v_0[idx * 3 + 0] * dt;
                x_curr[idx * 3 + 1] = x_0[idx * 3 + 1] + v_0[idx * 3 + 1] * dt;
                x_curr[idx * 3 + 2] = x_0[idx * 3 + 2] + v_0[idx * 3 + 2] * dt;
            }
            else {
                x_curr[idx * 3 + 0] = x_0[idx * 3 + 0];
                x_curr[idx * 3 + 1] = x_0[idx * 3 + 1];
                x_curr[idx * 3 + 2] = x_0[idx * 3 + 2];
            }
        }
        else {
            v_0[idx * 3 + 0] = 0;
            v_0[idx * 3 + 1] = 0;
            v_0[idx * 3 + 2] = 0;
            x_curr[idx * 3 + 0] = x_0[idx * 3 + 0];
            x_curr[idx * 3 + 1] = x_0[idx * 3 + 1];
            x_curr[idx * 3 + 2] = x_0[idx * 3 + 2];
        }
        ADU::Real m = M[idx];
        M_x_tilde[idx * 3 + 0] = m * x_curr[idx * 3 + 0];
        M_x_tilde[idx * 3 + 1] = m * x_curr[idx * 3 + 1];
        M_x_tilde[idx * 3 + 2] = m * x_curr[idx * 3 + 2];
    }
}

void do_pre_integration(int nVert, int nDynVert, const ADU::Real dt, const ADU::Real g, const ADU::Real* x_0, const int* is_fix, const ADU::Real* M,
    ADU::Real* v_0, ADU::Real* x_curr, ADU::Real* M_x_tilde)
{

    int blockSize = 256; // This can be adjusted
    int numBlocks = (nVert + blockSize - 1) / blockSize;
    // set v_0 to 0
    // get x_tilde
    // get M_x_tilde
    // 
    do_integration_kernel
    <<<numBlocks, blockSize>>>
    (nVert, nDynVert, dt, g, x_0, is_fix, M, v_0, x_curr, M_x_tilde);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

}


__global__ void do_post_process_kernel(int nVert, int nDynVert, const ADU::Real dt_inv, ADU::Real* x_0, 
    ADU::Real* v_0, const ADU::Real* x_curr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nVert) {
        if (idx < nDynVert) {
            v_0[idx * 3 + 0] = (x_curr[idx * 3 + 0] - x_0[idx * 3 + 0]) * dt_inv;
            v_0[idx * 3 + 1] = (x_curr[idx * 3 + 1] - x_0[idx * 3 + 1]) * dt_inv;
            v_0[idx * 3 + 2] = (x_curr[idx * 3 + 2] - x_0[idx * 3 + 2]) * dt_inv;
        }

        x_0[idx * 3 + 0] = x_curr[idx * 3 + 0];
        x_0[idx * 3 + 1] = x_curr[idx * 3 + 1];
        x_0[idx * 3 + 2] = x_curr[idx * 3 + 2];
    }
}

void do_post_process(int nVert, int nDynVert, const ADU::Real dt, ADU::Real* v_0, ADU::Real* x_curr, ADU::Real* x_0) 
{
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nVert + blockSize - 1) / blockSize;
    do_post_process_kernel
        << <numBlocks, blockSize >> >
    (nVert, nDynVert, 1.0f / dt, x_0, v_0, x_curr);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}