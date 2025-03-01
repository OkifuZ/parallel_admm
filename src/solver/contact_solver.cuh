#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "contact/collision_detection_cuda.h"


void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact);

void compute_Scc_impl_cu(const BVH_GPU* bvh, const thrust::device_vector<int>& d_vi_ct_nums, const thrust::device_vector<float4>& d_bary, const int nContact,
    thrust::device_vector<int>& d_start_idx, thrust::device_vector<int>& d_involved_cid, thrust::device_vector<ADU::Real>& d_Gamma_i,
    thrust::device_vector<float4>& d_Gamma_c, thrust::device_vector<float3>& d_K_c, thrust::device_vector<float3>& d_delta_u);