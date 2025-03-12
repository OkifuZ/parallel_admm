#pragma once


#include <Eigen/Core>

#include "src_config.h"

#include "mutils/exception_handle.h"
#include "mutils/cformat.h"
#include "mutils/common_types.h"
#include "mutils/timer.h"
#include "mutils/aux_color.h"
#include "mesh/mesh_container.h"

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
#include "mutils/dist_g.cuh"

#include "contact/narrow_phase.h"

//void Init_CUDA() {
//    cudaError_t cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        exit(0);
//    }
//}

class Solver;

class BVH_GPU {
public:

    size_t max_collision_number = 100000;
    size_t hist_max_collision_num = 0;

    lbvh_f bvh_f;
    lbvh_e bvh_e;
    std::vector<AABB> bvs;
    std::vector<Node> nodes;

    Solver* solver;
    MeshData* mesh;

    float3* d_verts;
    uint2* d_edges;
    uint3* d_faces;
    uint32_t* d_surfVertIdx;
    int4* d_collisonPairs; // data 1
    std::vector<int4> h_collisionPairs;


    size_t v_num_surf{};
    size_t v_num{};
    size_t e_num{};
    size_t t_num{};

    Result<ADU::Real>* d_contact_info; // data 2
    std::vector<Result<ADU::Real>> h_contact_info;
    uint32_t* d_cpNum;
    uint32_t h_cpNum; // data 3

    void init(size_t max_collision_number);

    void construct(bool copy_to_host = false);

    void update(const ADU::Matf_X3& verts, bool copy_to_host = false);

    void update(ADU::Real* verts_device, int vnum, bool copy_to_host = false);

    void convert_contactInfo_device2host(ProximalQuery::ContactInfoList& ct_info, const ADU::Matf_X3& pos);

    void unique_contactInfo();

    void dcd();

};