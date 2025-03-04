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
#include "parallel_solver.h"


using namespace  ADU;

// Addition
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Subtraction
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Scalar Multiplication
__host__ __device__ inline float3 operator*(const float3& a, float scalar) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__ inline float3 operator*(float scalar, const float3& a) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

// Dot Product
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float fnorm(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}


// Define a traits struct to infer the correct element type
template <typename T>
struct element_type;

// Specializations for float3 and float4
template <>
struct element_type<float3> {
    using type = float;
};

template <>
struct element_type<float4> {
    using type = float;
};

// Specialization for int4
template <>
struct element_type<int4> {
    using type = int;
};

// Universal function for retrieving the i-th element with automatic type deduction
template <typename T>
__host__ __device__ inline typename element_type<T>::type& ele(T& vec, int i) {
    return reinterpret_cast<typename element_type<T>::type*>(&vec)[i];
}

// Const version for read-only access
template <typename T>
__host__ __device__ inline const typename element_type<T>::type& ele(const T& vec, int i) {
    return reinterpret_cast<const typename element_type<T>::type*>(&vec)[i];
}


__host__ __device__ inline float3 load_float3(const ADU::Real* ptr) {
    return make_float3((float)ptr[0], (float)ptr[1], (float)ptr[2]);
}


__device__ inline void atomicAdd_F3(float3* address, const float3& value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
    atomicAdd(&(address->z), value.z);
}

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
    thrust::fill(d_vi_ct_nums.begin(), d_vi_ct_nums.end(), 0);
    //CUDA_SAFE_CALL(cudaMemset(thrust::raw_pointer_cast(d_vi_ct_nums.data()), 0, d_vi_ct_nums.size() * sizeof(int)));
    updateContactCounts << <numBlocks, blockSize >> > (d_contacts, thrust::raw_pointer_cast(d_vi_ct_nums.data()), nContact);
    cudaDeviceSynchronize(); // Ensure the kernel finishes

    /*std::vector<int> v_ct_num(d_vi_ct_nums.size(), 0);
    thrust::copy(d_vi_ct_nums.begin(), d_vi_ct_nums.end(), v_ct_num.begin());
    for (int i = 0; i < d_vi_ct_nums.size(); i++) {
        printf("%d ", v_ct_num[i]);
    }
    printf("\n");*/
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

        Vecf_3 normal = (pair_tp == 0) ? (v1 - v0).cross(v3 - v2).normalized() : (v2 - v1).cross(v3 - v1).normalized();
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
    ContactDataDevice* ct_data_device
) {
    int nContact = bvh->h_cpNum;

    ct_data_device->d_bary.resize(nContact, float4{});
    ct_data_device->d_normal.resize(nContact, float3{});
    ct_data_device->d_point.resize(nContact, float3{});
    ct_data_device->d_h_cN.resize(nContact, 0);
    ct_data_device->d_pair_type.resize(nContact, 0);

    /*thrust::fill(ct_data_device->d_bary.begin(), ct_data_device->d_bary.end(), float4{ 0 });
    thrust::fill(ct_data_device->d_normal.begin(), ct_data_device->d_normal.end(), float3{ 0 });
    thrust::fill(ct_data_device->d_point.begin(), ct_data_device->d_point.end(), float3{ 0 });
    thrust::fill(ct_data_device->d_h_cN.begin(), ct_data_device->d_h_cN.end(), 0);
    thrust::fill(ct_data_device->d_pair_type.begin(), ct_data_device->d_pair_type.end(), 0);*/

    // Launch kernel with a sufficient number of threads per block
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize;
    convert_DCD_info_kernel << <numBlocks, blockSize >> > 
        (
        nContact, bvh->d_contact_info, thrust::raw_pointer_cast(d_pos.data()), bvh->d_collisonPairs, ContactParameter::get_thickness(), 
            thrust::raw_pointer_cast(ct_data_device->d_bary.data()), thrust::raw_pointer_cast(ct_data_device->d_normal.data()), thrust::raw_pointer_cast(ct_data_device->d_point.data()),
            thrust::raw_pointer_cast(ct_data_device->d_h_cN.data()), thrust::raw_pointer_cast(ct_data_device->d_pair_type.data())
        );
}


__global__ void compute_Scc_kernel(const int4* d_contacts, const Result<ADU::Real>* d_contact_info, const int* d_start_idx, const float4* d_bary, const int* d_v_ct_nums, 
    const ADU::Real* d_contact_W_list, const ADU::Real* d_contact_W_inv_list,
    const ADU::Real* d_h_cN, const float3* d_noraml, const ADU::Real gamma, const ADU::Real epslon,
    const int nContact, const ADU::Real m_dt_inv,
    int* d_v_ct_count, // aux
    int* d_involved_cid, ADU::Real* d_Gamma_i,
    float4* d_Gamma_c, float3* d_K_c, float3* d_delta_u) {
    using namespace ADU;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nContact) {
        const auto& c = d_contacts[idx];
        const float4& bary = d_bary[idx];
        float Sc = epslon + 
            bary.x * bary.x  * d_contact_W_inv_list[c.x] * d_v_ct_nums[c.x] +
            bary.y * bary.y  * d_contact_W_inv_list[c.y] * d_v_ct_nums[c.y] +
            bary.z * bary.z * d_contact_W_inv_list[c.z] * d_v_ct_nums[c.z] +
            bary.w + bary.w * d_contact_W_inv_list[c.w] * d_v_ct_nums[c.w];
        
        printf("(%d %f) ", idx, Sc);
        /*printf("(%d %.8f %.8f %.8f %.8f) ", idx, d_contact_W_inv_list[c.x], 
            d_contact_W_inv_list[c.y], 
            d_contact_W_inv_list[c.z], 
            d_contact_W_inv_list[c.w]);*/
        //printf("(%d %d %d %d %d) ", idx, d_v_ct_nums[c.x], d_v_ct_nums[c.y], d_v_ct_nums[c.z], d_v_ct_nums[c.w]);
        //printf("%f %f %f %f\n", d_contact_W_list[c.x], d_contact_W_list[c.y], d_contact_W_list[c.z], d_contact_W_list[c.w]);
        //if (ele(c, 0) != c.x || ele(c, 0) != c.y || ele(c, 2) != c.z || ele(c, 3) != c.w) printf("herehit, \n");
        for (int vi = 0; vi < 4; vi++) {
            //int vind = *((int*)(&c) + vi);
            int vind = ele(c, vi);
            float w = d_contact_W_list[vind];
            int v_coffset = atomicAdd(&d_v_ct_count[vind], 1);
            int v_cidx = d_start_idx[vind] + v_coffset;
            //printf("%d; ", v_cidx);
            
            float4& gamma_c = d_Gamma_c[idx];
            
            //ele(gamma_c, vi) = ele(bary, vi) / (Sc * w);
            *((float*)(d_Gamma_c + idx) + vi) = ele(bary, vi) / (Sc * w);
            //*((float*)&(d_Gamma_c[idx]) + vi) = ele(bary, vi) / (Sc * w);
            // no need to check, 1e7 is large enough
            d_Gamma_i[v_cidx] = ele(bary, vi) / (Sc * w);
            //printf("(%f %f %f %f); ", d_Gamma_i[v_cidx], Sc, w, ele(bary, vi));
            //printf("(%f %f); ", d_contact_W_list[vind], d_contact_W_inv_list[vind]);
            d_involved_cid[v_cidx] = idx;
        }
       
        d_K_c[idx] = m_dt_inv * d_h_cN[idx] * d_noraml[idx] * gamma;
    }
}


void compute_Scc_impl_cu(const BVH_GPU* bvh, 
    const int nContact, const ADU::Real m_dt_inv, const ADU::Real gamma, const ADU::Real epslon,
    ContactDataDevice* ct_data_device)
{

    ct_data_device->d_Gamma_c.resize(nContact, float4{});
    ct_data_device->d_K_c.resize(nContact, float3{});
    ct_data_device->d_delta_u.resize(nContact, float3{});
    ct_data_device->d_r_c.resize(nContact, float3{});

    thrust::fill(ct_data_device->d_Gamma_c.begin(), ct_data_device->d_Gamma_c.end(), float4{ 0,0,0 ,0});
    thrust::fill(ct_data_device->d_K_c.begin(), ct_data_device->d_K_c.end(), float3{ 0,0,0 });
    thrust::fill(ct_data_device->d_delta_u.begin(), ct_data_device->d_delta_u.end(), float3{ 0,0,0 });
    thrust::fill(ct_data_device->d_r_c.begin(), ct_data_device->d_r_c.end(), float3{ 0,0,0 });

    ct_data_device->d_start_idx.resize(ct_data_device->d_vi_ct_nums.size() + 1, 0);
    thrust::fill(ct_data_device->d_start_idx.begin(), ct_data_device->d_start_idx.end(), 0);
    thrust::inclusive_scan(ct_data_device->d_vi_ct_nums.begin(), ct_data_device->d_vi_ct_nums.end(), ct_data_device->d_start_idx.begin() + 1);
    
    /*std::vector<int> v_ct_num(ct_data_device->d_vi_ct_nums.size(), 0);
    thrust::copy(ct_data_device->d_vi_ct_nums.begin(), ct_data_device->d_vi_ct_nums.end(), v_ct_num.begin());
    for (int i = 0; i < ct_data_device->d_vi_ct_nums.size(); i++) {
        printf("%d ", v_ct_num[i]);
    }
    printf("\n");*/
    int total_size{};
     total_size = ct_data_device->d_start_idx.back(); // this should work and indeed work, but use the following line for safety (mentally)
    //CUDA_SAFE_CALL(cudaMemcpy((void*)&total_size, thrust::raw_pointer_cast(ct_data_device->d_start_idx.data() + ct_data_device->d_start_idx.size() - 1), sizeof(int), cudaMemcpyDeviceToHost));
     //printf("total size: %d, ncontact: %d \n", total_size, nContact);

    ct_data_device->d_vi_ct_count.resize(ct_data_device->d_vi_ct_nums.size(), 0);
    ct_data_device->d_involved_cid.resize(total_size, 0);
    ct_data_device->d_Gamma_i.resize(total_size, 0);

    thrust::fill(ct_data_device->d_vi_ct_count.begin(), ct_data_device->d_vi_ct_count.end(), 0);
    thrust::fill(ct_data_device->d_involved_cid.begin(), ct_data_device->d_involved_cid.end(), 0);
    thrust::fill(ct_data_device->d_Gamma_i.begin(), ct_data_device->d_Gamma_i.end(), 0);


    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize; 
    compute_Scc_kernel
        << <numBlocks, blockSize >> > 
        (
        bvh->d_collisonPairs, bvh->d_contact_info, 
        thrust::raw_pointer_cast(ct_data_device->d_start_idx.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_bary.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_vi_ct_nums.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_contact_W_list.data()),
            thrust::raw_pointer_cast(ct_data_device->d_contact_W_inv_list.data()),
            thrust::raw_pointer_cast(ct_data_device->d_h_cN.data()),
            thrust::raw_pointer_cast(ct_data_device->d_normal.data()), gamma, epslon,
        nContact, m_dt_inv, 
        thrust::raw_pointer_cast(ct_data_device->d_vi_ct_count.data()),
        thrust::raw_pointer_cast(ct_data_device->d_involved_cid.data()), thrust::raw_pointer_cast(ct_data_device->d_Gamma_i.data()),
        thrust::raw_pointer_cast(ct_data_device->d_Gamma_c.data()), thrust::raw_pointer_cast(ct_data_device->d_K_c.data()), thrust::raw_pointer_cast(ct_data_device->d_delta_u.data())
        );

    /*std::vector<int> iv_cid(total_size, 0);
    std::vector<int> st_cid(ct_data_device->d_start_idx.size(), 0);

    thrust::copy(ct_data_device->d_involved_cid.begin(), ct_data_device->d_involved_cid.end(), iv_cid.begin());
    thrust::copy(ct_data_device->d_start_idx.begin(), ct_data_device->d_start_idx.end(), st_cid.begin());
    for (int i = 0; i < st_cid.size() - 1; i++) {
        int st = st_cid[i];
        int ed = st_cid[i+1];
        for (int i = st; i < ed; i++) {
            printf("%d ", iv_cid[i]);
        }
        printf(", ");
    }
    printf("\n");*/

   /* std::vector<int> v_ct_count(ct_data_device->d_vi_ct_nums.size(), 0);
    std::vector<int> v_ct_num(ct_data_device->d_vi_ct_nums.size(), 0);
    thrust::copy(ct_data_device->d_vi_ct_count.begin(), ct_data_device->d_vi_ct_count.end(), v_ct_count.begin());
    thrust::copy(ct_data_device->d_vi_ct_nums.begin(), ct_data_device->d_vi_ct_nums.end(), v_ct_num.begin());
    for (int i = 0; i < v_ct_count.size(); i++) {
        if (v_ct_num[i] != v_ct_count[i]) {
            printf("%d %d\n", v_ct_count[i], v_ct_num[i]);
            printf("error compute_Scc_impl_cu involve_cid! vid %d\n", i);
        }
    }*/
}


__global__ void update_p_kernel(const int nContact, const int4* d_contact, const float4* d_Gamma_c, ADU::Real* d_p, float3* d_r_c, int static_vert_begin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nContact) 
    {
        const int4& ct = d_contact[idx];
        const float4& gamma_c = d_Gamma_c[idx];
        for (int vid = 0; vid < 4; vid++) {
            int vind = ele(ct, vid);  
            if (vind < static_vert_begin) {
                float gamma_cvi = ele(gamma_c, vid);
                float3 r_c = gamma_cvi * d_r_c[idx];
                atomicAdd(&d_p[vind * 3 + 0], r_c.x);
                atomicAdd(&d_p[vind * 3 + 1], r_c.y);
                atomicAdd(&d_p[vind * 3 + 2], r_c.z);
            }
        }
    }
}

__global__ void get_delta_u_kernel(const int nContact, const ADU::Real mu,
    const int4* d_contact, const float4* d_bary, const ADU::Real* d_p, const float3* d_K_c, const float3* d_normal,
    float3* d_delta_u, float3* d_r_c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nContact)
    {
        float3 u{};
        const int4& ct = d_contact[idx];
        const float4& bary = d_bary[idx];
        //int vinds[4]{ ct.x, ct.y, ct.z, ct.w };
        u = bary.x * load_float3(d_p + 3 * ct.x) +
            bary.y * load_float3(d_p + 3 * ct.y) +
            bary.z * load_float3(d_p + 3 * ct.z) +
            bary.w * load_float3(d_p + 3 * ct.w);
        //printf("[%f %f %f %f], ", bary.x, bary.y, bary.z, bary.w);
        u = u + d_K_c[idx];
        const float3& normal = d_normal[idx];

        float3 u_star = u - d_r_c[idx];
        float u_star_N = dot(normal, u_star);
        //printf("%f, ", u_star_N);
        
        float3 u_star_T = u_star - u_star_N * normal;
        //printf("(%f %f %f; %f %f %f; %f %f %f)", normal.x, normal.y, normal.z, u.x, u.y, u.z, d_r_c[idx].x, d_r_c[idx].y, d_r_c[idx].z);

        float tau, alpha;
        if (u_star_N < 0) {
            tau = fnorm(u_star_T);
            alpha = -mu * u_star_N;
            u_star_N = 0;
            /*if (tau <= alpha) {
                u_star_T = float3{ 0,0,0 };
            }
            else {
                u_star_T = (1.0f - alpha / tau) * u_star_T;
            }*/
        }

        float3 u_star_new = u_star_N * normal + u_star_T;
        d_delta_u[idx] = u_star_new - u;
        d_r_c[idx] = d_r_c[idx] + d_delta_u[idx];
        //printf("(%f %f %f)", d_delta_u[idx].x, d_delta_u[idx].y, d_delta_u[idx].z);

    }
}

__global__ void update_p(const int nVert, const int* involved_cid, const int* start, const ADU::Real* gamma_i, const float3* delta_u, ADU::Real* d_p) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nVert)
    {
        float3 delta_pi{ 0,0,0 };
        int start_idx = start[idx];
        int end_idx = start[idx + 1]; 
        //if (start_idx != end_idx) printf("%d: %d %d, ", idx, start_idx, end_idx); // empty BUG!
        for (int i = start_idx; i < end_idx; i++) {
            int cid = involved_cid[i];
            delta_pi = delta_pi + delta_u[cid] * gamma_i[i];
            //printf("%f, ", gamma_i[i]);
        }
        d_p[idx * 3 + 0] += delta_pi.x;
        d_p[idx * 3 + 1] += delta_pi.y;
        d_p[idx * 3 + 2] += delta_pi.z;
    }
    
}


void project_impl(const BVH_GPU* bvh, ContactDataDevice* ct_data_device, CuSolverData* solver_data, int max_iter, int static_vert_begin, ADU::Real mu, int nVert)
{
    int nContact = bvh->h_cpNum;
    // kernel 1: p + r_c
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize;
    update_p_kernel 
        << <numBlocks, blockSize >> > 
        (nContact, bvh->d_collisonPairs, thrust::raw_pointer_cast(ct_data_device->d_Gamma_c.data()),
        thrust::raw_pointer_cast(solver_data->p_device.data()), thrust::raw_pointer_cast(ct_data_device->d_r_c.data()), static_vert_begin);

    for (int i = 0; i < max_iter; i++) {
        // kernel 2: delta_u
        get_delta_u_kernel
        << <numBlocks, blockSize >> >
        (nContact, mu, bvh->d_collisonPairs, thrust::raw_pointer_cast(ct_data_device->d_bary.data()), 
            thrust::raw_pointer_cast(solver_data->p_device.data()), thrust::raw_pointer_cast(ct_data_device->d_K_c.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_normal.data()), thrust::raw_pointer_cast(ct_data_device->d_delta_u.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_r_c.data()));

        // kernel 3: update p
        blockSize = 256; // This can be adjusted
        int numBlocks = (nVert + blockSize - 1) / blockSize;
        update_p
            << <numBlocks, blockSize >> >
        (nVert, thrust::raw_pointer_cast(ct_data_device->d_involved_cid.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_start_idx.data()), thrust::raw_pointer_cast(ct_data_device->d_Gamma_i.data()), 
            thrust::raw_pointer_cast(ct_data_device->d_delta_u.data()), thrust::raw_pointer_cast(solver_data->p_device.data()));
    }
}