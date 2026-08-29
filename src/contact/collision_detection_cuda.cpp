#include "contact/collision_detection_cuda.h"


#include <Eigen/Core>

#include "src_config.h"
#include "solver/solver.h"

#include "mutils/exception_handle.h"
#include "mutils/cformat.h"
#include "mutils/common_types.h"
#include "mutils/timer.h"
#include "mutils/aux_color.h"

#include <string>
#include <filesystem>
#include <memory>
#include <unordered_set>

#include <fstream>
#include <functional>

#include "mutils/common_types.h"
#include "mutils/cuda_tools.cuh"
#include "device_launch_parameters.h"
#include "contact/lbvh/lbvh.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/detail/unique.inl>

#include "mutils/dist_g.cuh"
#include "contact/contact_util.cuh"

#include "solver/solver.h"



void BVH_GPU::init(size_t max_collision_number) {
    this->max_collision_number = max_collision_number;
    auto verts = solver->getVertices(); //
    auto edges = mesh->surface_edges;
    auto tris = mesh->surface_tris;

    this->v_num_surf = mesh->surface_vinds.rows();
    this->v_num = verts.rows();
    this->e_num = edges.rows();
    this->t_num = tris.rows();

    //Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_double = verts.cast<double>();

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_verts, v_num * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_faces, t_num * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edges, e_num * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_surfVertIdx, v_num_surf * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_contact_info, max_collision_number * sizeof(Result<ADU::Real>)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_collisonPairs, max_collision_number * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cpNum, sizeof(uint32_t)));

    h_contact_info.reserve(max_collision_number);
    h_collisionPairs.reserve(max_collision_number);

    std::vector<uint32_t> h_surfVertIdx(v_num_surf);
    for (int i = 0; i < v_num_surf; i++) {
        h_surfVertIdx[i] = mesh->surface_vinds(i);
    }
    /*std::iota(h_surfVertIdx.begin(), h_surfVertIdx.end(), 0);*/
    
    CUDA_SAFE_CALL(cudaMemcpy((void*)d_surfVertIdx, h_surfVertIdx.data(), v_num_surf * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)d_faces, tris.data(), t_num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)d_edges, edges.data(), e_num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)d_verts, verts.data(), v_num * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemset((void*)d_cpNum, 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset((void*)d_contact_info, 0, max_collision_number * sizeof(Result<ADU::Real>)));
    CUDA_SAFE_CALL(cudaMemset((void*)d_collisonPairs, 0, max_collision_number * sizeof(int4)));

    bvh_f.init(nullptr, d_verts, d_faces, d_surfVertIdx, d_collisonPairs, d_cpNum, t_num, v_num_surf, d_contact_info);
    bvh_e.init(nullptr, d_verts, nullptr, d_edges, d_collisonPairs, d_cpNum, e_num, v_num_surf, d_contact_info);
    bvh_e.face_number = t_num;
    size_t bvs_max_size = 2 * this->t_num - 1 + 2 * this->e_num - 1 + 100000;
    bvs.reserve(bvs_max_size);
}


void BVH_GPU::construct(bool copy_to_host) {
    bvh_e.Construct();
    bvh_f.Construct();

    int t_bvh_size = 2 * this->t_num - 1;
    int e_bvh_size = 2 * this->e_num - 1;

    if (copy_to_host) {
        bvs.resize(2 * this->t_num - 1 + 2 * this->e_num - 1);
        CUDA_SAFE_CALL(cudaMemcpy(&bvs[0], bvh_f._bvs, t_bvh_size * sizeof(AABB), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&bvs[t_bvh_size], bvh_e._bvs, e_bvh_size * sizeof(AABB), cudaMemcpyDeviceToHost));
    }
}

void BVH_GPU::update(const ADU::Matf_X3& verts, bool copy_to_host) {
    //auto verts = solver->getVertices();
    const auto& edges = mesh->surface_edges;
    const auto& tris = mesh->surface_tris;

    this->v_num = v_num;
    this->e_num = edges.rows();
    this->t_num = tris.rows();

    //Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> verts_double = verts.cast<double>();

    //CUDA_SAFE_CALL(cudaMemcpy(d_verts, verts_double.data(), v_num * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_verts, verts.data(), v_num * sizeof(float3), cudaMemcpyHostToDevice));

    bvh_f._vertexes = d_verts;
    bvh_e._vertexes = d_verts;
    //bvh_f.init(nullptr, d_verts, d_faces, d_surfVertIdx, d_collisonPairs, d_cpNum, t_num, v_num, d_contact_info);
    //bvh_e.init(nullptr, d_verts, nullptr, d_edges, d_collisonPairs, d_cpNum, e_num, v_num, d_contact_info);

    // TODO init got memory leak!

    construct(copy_to_host);
}



void BVH_GPU::update(ADU::Real* verts_device, int vnum, bool copy_to_host) {
    //auto verts = solver->getVertices();
    const auto& edges = mesh->surface_edges;
    const auto& tris = mesh->surface_tris;

    this->v_num = vnum;
    this->e_num = edges.rows();
    this->t_num = tris.rows();
    d_verts = reinterpret_cast<typename ADU::ADU_ele_traits<ADU::Real>::vec3_type*>(verts_device);

    bvh_f._vertexes = d_verts;
    bvh_e._vertexes = d_verts;
    //bvh_f.init(nullptr, d_verts, d_faces, d_surfVertIdx, d_collisonPairs, d_cpNum, t_num, v_num, d_contact_info);
    //bvh_e.init(nullptr, d_verts, nullptr, d_edges, d_collisonPairs, d_cpNum, e_num, v_num, d_contact_info);

    // TODO init got memory leak!

    construct(copy_to_host);
}

void BVH_GPU::dcd() {

    CUDA_SAFE_CALL(cudaMemset(d_cpNum, 0, sizeof(uint32_t)));
    this->bvh_f.discreteCollisionDetection(ContactParameter::get_thickness(), v_num_surf);
    this->bvh_e.discreteCollisionDetection(ContactParameter::get_thickness());
    CUDA_SAFE_CALL(cudaMemcpy(&h_cpNum, d_cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("cp nums: %d\n", h_cpNum);
    hist_max_collision_num = std::max(hist_max_collision_num, (size_t)h_cpNum);
}

void BVH_GPU::unique_contactInfo() {
    size_t new_size = removeDuplicates(d_collisonPairs, d_contact_info, h_cpNum);

    if (new_size != h_cpNum) {
        printf("new size = %d, old size = %d\n", new_size, h_cpNum);
    }

    int less_num = h_cpNum - new_size;
    printf("unique: less %d\n", less_num);
    h_cpNum = new_size;


    CUDA_SAFE_CALL(cudaMemcpy(d_cpNum, &h_cpNum, sizeof(uint32_t), cudaMemcpyHostToDevice));
}



void BVH_GPU::download_contact_info() {
    h_contact_info.resize(h_cpNum);
    if (h_cpNum > 0) {
        CUDA_SAFE_CALL(cudaMemcpy((void*)h_contact_info.data(), (void*)d_contact_info, h_cpNum * sizeof(Result<float>), cudaMemcpyDeviceToHost));
    }
}

std::vector<std::array<float, 3>> BVH_GPU::points() {
    download_contact_info();
    std::vector<std::array<float, 3>> pts(h_cpNum);
    for (size_t i = 0; i < pts.size(); i++) {
        const auto& res = h_contact_info[i];
        const ADU::Vecf_3 mid = (res.closest[0] + res.closest[1]) * static_cast<ADU::Real>(0.5);
        pts[i] = { mid.x(), mid.y(), mid.z() };
    }
    return pts;
}

std::vector<std::array<float, 3>> BVH_GPU::normals() {
    download_contact_info();
    std::vector<std::array<float, 3>> normals(h_cpNum);
    for (size_t i = 0; i < normals.size(); i++) {
        const auto& res = h_contact_info[i];
        ADU::Vecf_3 n = res.closest[0] - res.closest[1];
        const ADU::Real len = n.norm();
        if (len > 0) n /= len;
        normals[i] = { n.x(), n.y(), n.z() };
    }
    return normals;
}

void BVH_GPU::convert_contactInfo_device2host(ProximalQuery::ContactInfoList& ct_info, const ADU::Matf_X3& pos) {
    using namespace ADU;
    h_contact_info.resize(h_cpNum);
    h_collisionPairs.resize(h_cpNum);

    CUDA_SAFE_CALL(cudaMemcpy((void*)h_contact_info.data(), (void*)d_contact_info, h_cpNum * sizeof(Result<float>), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy((void*)h_collisionPairs.data(), (void*)d_collisonPairs, h_cpNum * sizeof(int4), cudaMemcpyDeviceToHost));


    ct_info.resize(h_cpNum);
    std::cout << "h_cpNum: " << h_cpNum << std::endl;
    if (h_cpNum > 0) {
        if (h_cpNum == 15) {
            printf("hit\n");
        }
        for (int i = 0; i < h_cpNum; i++) {
            const Result<ADU::Real>& res = h_contact_info[i];
            int4& inds = h_collisionPairs[i];
            // point
            const Vecf_3& v0 = pos.row(inds.x);
            // triangle
            const Vecf_3& v1 = pos.row(inds.y);
            const Vecf_3& v2 = pos.row(inds.z);
            const Vecf_3& v3 = pos.row(inds.w);

            /*if (inds.x >= 268 || inds.y >= 268 || inds.z >= 268 || inds.w >= 268 ||
                inds.x < 0|| inds.y < 0|| inds.z < 0|| inds.w < 0) {
                printf("hit %d\n", i);
            }*/
            // printf("%d %d %d %d\n", inds.x, inds.y, inds.z, inds.w);

            size_t v0_index = inds.x;
            size_t v1_index = inds.y;
            size_t v2_index = inds.z;
            size_t v3_index = inds.w;

            if (res.barycentric[2] < -1) { // edge edge
                Vecf_3 normal = (v1 - v0).cross(v3 - v2).normalized();
                Vecf_3 pt = 0.5_r * (res.closest[0].cast<ADU::Real>() + res.closest[1].cast<ADU::Real>());
                //Vecf_3 pt = res.closest[0].cast<ADU::Real>();
                ContactInfo ct(res, v0_index, v1_index, v2_index, v3_index, normal, pt, false);
                ct_info[i] = ct;
            }
            else { // point traingle
                Vecf_3 normal = (v2 - v1).cross(v3 - v1).normalized();
                Vecf_3 pt = 0.5_r * (res.closest[0].cast<ADU::Real>() + res.closest[1].cast<ADU::Real>());
                ContactInfo ct(res, v0_index, v1_index, v2_index, v3_index, normal, pt, false);
                ct_info[i] = ct;
            }
        }
    }
}



