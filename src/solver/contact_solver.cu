#include <cuda_runtime.h>
#include "mutils/cuda_tools.cuh"
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include "mutils/common_types.h"
#include "mutils/dist_g.cuh"

#include "solver/contact_solver.cuh"


using namespace  ADU;

__global__ void updateContactCounts(const int4* contacts, int* vi_ct_nums, int nContact) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nContact) {
        const auto& c = contacts[idx];

        // Use atomicAdd to update the counts for each vertex index
        atomicAdd(&vi_ct_nums[c.x], 1);
        atomicAdd(&vi_ct_nums[c.y], 1);
        atomicAdd(&vi_ct_nums[c.z], 1);
        atomicAdd(&vi_ct_nums[c.w], 1);
    }
}

void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact) {

    // Launch kernel with a sufficient number of threads per block
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize;
    CUDA_SAFE_CALL(cudaMemset(thrust::raw_pointer_cast(d_vi_ct_nums.data()), 0, d_vi_ct_nums.size() * sizeof(int)));
    updateContactCounts << <numBlocks, blockSize >> > (d_contacts, thrust::raw_pointer_cast(d_vi_ct_nums.data()), nContact);
    cudaDeviceSynchronize(); // Ensure the kernel finishes
}


__global__ void compute_Scc_kernel(const int4* d_contacts, const Result<ADU::Real>* d_contact_info, const int* d_start_idx, const int nContact, 
    int* d_v_ct_count, // aux
    int* d_involved_cid, ADU::Real* d_Gamma_i,
    float4* d_Gamma_c, float3* d_K_c, float3* d_delta_u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nContact) {
        const auto& c = d_contacts[idx];

        const int v_cidx_1 = atomicAdd(&d_v_ct_count[c.x], 1);
        d_involved_cid[d_start_idx[c.x] + v_cidx_1] = idx;

        const int v_cidx_2 = atomicAdd(&d_v_ct_count[c.y], 1);
        d_involved_cid[d_start_idx[c.y] + v_cidx_2] = idx;
        
        const int v_cidx_3 = atomicAdd(&d_v_ct_count[c.z], 1);
        d_involved_cid[d_start_idx[c.z] + v_cidx_3] = idx;
        
        const int v_cidx_4 = atomicAdd(&d_v_ct_count[c.w], 1);
        d_involved_cid[d_start_idx[c.w] + v_cidx_4] = idx;
    
    }
}

void compute_Scc_impl_cu(const BVH_GPU* bvh, const thrust::device_vector<int>& d_vi_ct_nums, const  thrust::device_vector<float4> & d_bary, const int nContact,
    thrust::device_vector<int>& d_start_idx, thrust::device_vector<int>& d_involved_cid, thrust::device_vector<ADU::Real>& d_Gamma_i, 
    thrust::device_vector<float4>& d_Gamma_c, thrust::device_vector<float3>& d_K_c, thrust::device_vector<float3>& d_delta_u)
{

    d_Gamma_c.resize(nContact, float4{});
    d_K_c.resize(nContact, float3{});
    d_delta_u.resize(nContact, float3{});

    d_start_idx.resize(d_vi_ct_nums.size() + 1, 0);
    thrust::inclusive_scan(d_vi_ct_nums.begin(), d_vi_ct_nums.end(), d_start_idx.begin() + 1);
    int total_size{};
    // total_size = d_start_idx.back(); // this should work and indeed work, but use the following line for safety (mentally)
    CUDA_SAFE_CALL(cudaMemcpy((void*)&total_size, thrust::raw_pointer_cast(d_start_idx.data() + d_start_idx.size() - 1), sizeof(int), cudaMemcpyDeviceToHost));
    d_involved_cid.resize(total_size, 0);

    thrust::device_vector<int> d_v_ct_count(d_vi_ct_nums.size(), 0);

    d_Gamma_i.resize(total_size, 0);

    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize; 
    compute_Scc_kernel << <numBlocks, blockSize >> > (
        bvh->d_collisonPairs, bvh->d_contact_info, thrust::raw_pointer_cast(d_start_idx.data()), nContact,
        thrust::raw_pointer_cast(d_v_ct_count.data()), 
        thrust::raw_pointer_cast(d_involved_cid.data()), thrust::raw_pointer_cast(d_Gamma_i.data()), 
        thrust::raw_pointer_cast(d_Gamma_c.data()), thrust::raw_pointer_cast(d_K_c.data()), thrust::raw_pointer_cast(d_delta_u.data())
        );

    std::vector<int> v_ct_count(d_vi_ct_nums.size());
    std::vector<int> v_ct_num(d_vi_ct_nums.size());
    thrust::copy(d_v_ct_count.begin(), d_v_ct_count.end(), v_ct_count.begin());
    thrust::copy(d_vi_ct_nums.begin(), d_vi_ct_nums.end(), v_ct_num.begin());
    for (int i = 0; i < v_ct_count.size(); i++) {
        if (v_ct_num[i] != v_ct_count[i]) printf("error compute_Scc_impl_cu involve_cid! vid %d\n", i);
    }
}