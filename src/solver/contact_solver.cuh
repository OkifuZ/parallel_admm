#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "contact/collision_detection_cuda.h"

class ContactDataDevice;
class CuSolverData;

void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact, int nVert);

void compute_Scc_impl_cu(const BVH_GPU* bvh,
    const int nContact, const ADU::Real m_dt_inv, const ADU::Real gamma, const ADU::Real epslon,
    ContactDataDevice* ct_data_device);

void convert_DCD_info(const BVH_GPU* bvh, 
    const thrust::device_vector<ADU::Real>& d_pos,
    ContactDataDevice* ct_data_device
);
// float4: bary
// float3: nomral, point
// float: h_cN, dist
// bool: pair_type


void project_impl(const BVH_GPU* bvh, ContactDataDevice* ct_data_device, CuSolverData* solver_data, int max_iter, int static_vert_begin, ADU::Real mu, int nVert);