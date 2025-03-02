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

void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact, int nVert) {

    // Launch kernel with a sufficient number of threads per block
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize;
    d_vi_ct_nums.resize(nVert, 0);
    //CUDA_SAFE_CALL(cudaMemset(thrust::raw_pointer_cast(d_vi_ct_nums.data()), 0, d_vi_ct_nums.size() * sizeof(int)));
    updateContactCounts << <numBlocks, blockSize >> > (d_contacts, thrust::raw_pointer_cast(d_vi_ct_nums.data()), nContact);
    cudaDeviceSynchronize(); // Ensure the kernel finishes
}



__global__ void convert_DCD_info_kernel(const int nContact, const Result<ADU::Real>* d_contact_info, const ADU::Real* pos, const int4* d_contacts, const ADU::Real thickness, 
    float4* d_bary, float3* d_normal, float3* d_point, ADU::Real* d_h_cN, int* pair_type)
{
    using namespace ADU;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nContact) 
    {
        int4 vids = d_contacts[idx];
        Vecf_3 v0{ pos[vids.x * 3 + 0], pos[vids.x * 3 + 1] , pos[vids.x * 3 + 2] };
        Vecf_3 v1{ pos[vids.y * 3 + 0], pos[vids.y * 3 + 1] , pos[vids.y * 3 + 2] };
        Vecf_3 v2{ pos[vids.z * 3 + 0], pos[vids.z * 3 + 1] , pos[vids.z * 3 + 2] };
        Vecf_3 v3{ pos[vids.w * 3 + 0], pos[vids.w * 3 + 1] , pos[vids.w * 3 + 2] };

        auto& res = d_contact_info[idx];
        Vecf_3 pt = 0.5f * (res.closest[0] + res.closest[1]);
        d_point[idx] = float3{ (float)pt.x(), (float)pt.y(), (float)pt.z() };
        
        int pair_tp = (res.barycentric[2] < -1) ? 0 : 1; // 0: ee, 1: ft
        pair_type[idx] = pair_tp;

        d_bary[idx] = pair_tp == 0 ? 
            float4{ 1 - (float)res.barycentric[0], (float)res.barycentric[0], (float)res.barycentric[1] - 1, -(float)res.barycentric[1] } :
            float4{ 1, -(float)res.barycentric[0], -(float)res.barycentric[1], -(float)res.barycentric[2] };

        Vecf_3 normal = (res.barycentric[2] < -1) ? (v1 - v0).cross(v3 - v2).normalized() : normal = (v2 - v1).cross(v3 - v1).normalized();
        Vecf_3 h = res.closest[0] - res.closest[1];
        Real h_CN = h.dot(normal);
        if (h_CN < 0) {
            normal *= -1;
            h_CN = -h_CN;
        }

        d_h_cN[idx] = h_CN - thickness;
        d_normal[idx] = float3{ (float)normal.x(), (float)normal.y(), (float)normal.z()};
    }
}



void convert_DCD_info(const BVH_GPU* bvh,
    const thrust::device_vector<ADU::Real>& d_pos, 
    thrust::device_vector<float4>& d_bary,
    thrust::device_vector<float3>& d_normal, thrust::device_vector<float3>& d_point,
    thrust::device_vector<ADU::Real>& d_h_cN, thrust::device_vector<int>& d_pair_type
) {
    int nContact = bvh->h_cpNum;

    d_bary.resize(nContact, float4{});
    d_normal.resize(nContact, float3{});
    d_point.resize(nContact, float3{});
    d_h_cN.resize(nContact, 0);
    d_pair_type.resize(nContact, 0);

    
    // Launch kernel with a sufficient number of threads per block
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize;
    convert_DCD_info_kernel << <numBlocks, blockSize >> > 
        (
        nContact, bvh->d_contact_info, thrust::raw_pointer_cast(d_pos.data()), bvh->d_collisonPairs, ContactParameter::get_thickness(), 
            thrust::raw_pointer_cast(d_bary.data()), thrust::raw_pointer_cast(d_normal.data()), thrust::raw_pointer_cast(d_point.data()), 
            thrust::raw_pointer_cast(d_h_cN.data()), thrust::raw_pointer_cast(d_pair_type.data())
        );
}


__global__ void compute_Scc_kernel(const int4* d_contacts, const Result<ADU::Real>* d_contact_info, const int* d_start_idx, const float4* d_bary, const int* d_v_ct_nums, const ADU::Real* d_contact_W_list, const ADU::Real* d_h_cN,
    const int nContact, const ADU::Real m_dt_inv,
    int* d_v_ct_count, // aux
    int* d_involved_cid, ADU::Real* d_Gamma_i,
    float4* d_Gamma_c, float3* d_K_c, float3* d_delta_u) {
    using namespace ADU;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nContact) {
        const auto& c = d_contacts[idx];
        const float4& bary = d_bary[idx];
        Real Sc = 
            bary.x * bary.x * d_v_ct_nums[c.x] * d_contact_W_list[c.x] +
            bary.y * bary.y * d_v_ct_nums[c.y] * d_contact_W_list[c.y] +
            bary.z * bary.z * d_v_ct_nums[c.z] * d_contact_W_list[c.z] +
            bary.w + bary.w * d_v_ct_nums[c.w] * d_contact_W_list[c.w];

        for (int vi = 0; vi < 4; vi++) {
            int vind = *((int*)(&c) + vi);
            ADU::Real  w = d_contact_W_list[vind];
            int v_coffset = atomicAdd(&d_v_ct_count[vind], 1);
            int v_cidx = d_start_idx[vind] + v_coffset;
            
            *((float*)&(d_Gamma_c[idx]) + vi) = bary.x / (Sc * w);
            // no need to check, 1e7 is large enough
            d_Gamma_i[v_cidx] = bary.x / (Sc * w);
            d_involved_cid[v_cidx] = idx;
        }
       
        /*const int v_cidx_1 = atomicAdd(&d_v_ct_count[c.x], 1); d_involved_cid[d_start_idx[c.x] + v_cidx_1] = idx;
        const int v_cidx_2 = atomicAdd(&d_v_ct_count[c.y], 1); d_involved_cid[d_start_idx[c.y] + v_cidx_2] = idx;
        const int v_cidx_3 = atomicAdd(&d_v_ct_count[c.z], 1); d_involved_cid[d_start_idx[c.z] + v_cidx_3] = idx;
        const int v_cidx_4 = atomicAdd(&d_v_ct_count[c.w], 1); d_involved_cid[d_start_idx[c.w] + v_cidx_4] = idx;*/

        // TODO K_c
    }
}


void compute_Scc_impl_cu(const BVH_GPU* bvh, const thrust::device_vector<int>& d_vi_ct_nums, const  thrust::device_vector<float4>& d_bary, const  thrust::device_vector<ADU::Real>& d_contact_W_list, const thrust::device_vector<ADU::Real>& d_h_cN,
    const int nContact, const ADU::Real m_dt_inv,
    thrust::device_vector<int>& d_vi_ct_count,
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

    d_vi_ct_count.resize(d_vi_ct_nums.size(), 0);
    d_involved_cid.resize(total_size, 0);
    d_Gamma_i.resize(total_size, 0);

    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize; 
    compute_Scc_kernel
        << <numBlocks, blockSize >> > 
        (
        bvh->d_collisonPairs, bvh->d_contact_info, thrust::raw_pointer_cast(d_start_idx.data()), thrust::raw_pointer_cast(d_bary.data()), thrust::raw_pointer_cast(d_vi_ct_nums.data()), thrust::raw_pointer_cast(d_contact_W_list.data()), thrust::raw_pointer_cast(d_h_cN.data()),
        nContact, m_dt_inv,
        thrust::raw_pointer_cast(d_vi_ct_count.data()), 
        thrust::raw_pointer_cast(d_involved_cid.data()), thrust::raw_pointer_cast(d_Gamma_i.data()), 
        thrust::raw_pointer_cast(d_Gamma_c.data()), thrust::raw_pointer_cast(d_K_c.data()), thrust::raw_pointer_cast(d_delta_u.data())
        );

    std::vector<int> v_ct_count(d_vi_ct_nums.size(), 0);
    std::vector<int> v_ct_num(d_vi_ct_nums.size(), 0);
    thrust::copy(d_vi_ct_count.begin(), d_vi_ct_count.end(), v_ct_count.begin());
    thrust::copy(d_vi_ct_nums.begin(), d_vi_ct_nums.end(), v_ct_num.begin());
    for (int i = 0; i < v_ct_count.size(); i++) {
        if (v_ct_num[i] != v_ct_count[i]) {
            printf("%d %d\n", v_ct_count[i], v_ct_num[i]);
            printf("error compute_Scc_impl_cu involve_cid! vid %d\n", i);
        }
    }
}