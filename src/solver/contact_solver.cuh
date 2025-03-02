#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "contact/collision_detection_cuda.h"


void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact, int nVert);

void compute_Scc_impl_cu(const BVH_GPU* bvh, const thrust::device_vector<int>& d_vi_ct_nums, const  thrust::device_vector<float4>& d_bary, const thrust::device_vector<ADU::Real>& d_contact_W_list, const thrust::device_vector<ADU::Real>& d_h_cN,
    const int nContact, const ADU::Real m_dt_inv,
    thrust::device_vector<int>& d_vi_ct_count,
    thrust::device_vector<int>& d_start_idx, thrust::device_vector<int>& d_involved_cid, thrust::device_vector<ADU::Real>& d_Gamma_i,
    thrust::device_vector<float4>& d_Gamma_c, thrust::device_vector<float3>& d_K_c, thrust::device_vector<float3>& d_delta_u);

void convert_DCD_info(const BVH_GPU* bvh, 
    const thrust::device_vector<ADU::Real>& d_pos,
    thrust::device_vector<float4>& d_bary,
    thrust::device_vector<float3>& d_normal, thrust::device_vector<float3>& d_point,
    thrust::device_vector<ADU::Real>& d_h_cN, thrust::device_vector<int>& pair_type
);
// float4: bary
// float3: nomral, point
// float: h_cN, dist
// bool: pair_type