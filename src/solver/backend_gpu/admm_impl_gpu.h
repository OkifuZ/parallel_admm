#pragma once

/// Phase 3.4: GPU ADMM implementation (migrated from parallel_solver.h).
/// Logic lives here; ADMMBackendGPU holds this via composition.

#include "solver/backend_cpu/admm_impl_cpu.h"
#include "constraint/bending_constraint_device.cuh"
#include "constraint/tetrahedral_constraint_device.cuh"

#include <thrust/device_vector.h>
#include <tbb/concurrent_vector.h>

class ContactDataDevice {
public:
    thrust::device_vector<int> d_vi_ct_nums;
    thrust::device_vector<int> d_vi_ct_count;
    thrust::device_vector<float4> d_bary;
    thrust::device_vector<float3> d_normal;
    thrust::device_vector<float3> d_point;
    thrust::device_vector<ADU::Real> d_h_cN;
    thrust::device_vector<int> d_pair_type;
    thrust::device_vector<float4> d_Gamma_c;
    thrust::device_vector<float3> d_K_c;
    thrust::device_vector<float3> d_delta_u;
    thrust::device_vector<float3> d_r_c;
    thrust::device_vector<ADU::Real> d_Gamma_i;
    thrust::device_vector<int> d_start_idx;
    thrust::device_vector<int> d_involved_cid;
    thrust::device_vector<ADU::Real> d_contact_W_list;
    thrust::device_vector<ADU::Real> d_contact_W_inv_list;

    ContactDataDevice(int max_Vert, int max_Contact);
};

namespace ADU {

/// GPU ADMM implementation (migrated from ADMMParallelSolver).
class ADMMImplGPU : public ADMMImplCPU {
public:
    void convert_constraint2device();

    void init() override;
    void precompute() override;
    void reset(const Matf_X3& ini_verts, bool need_precompute = false) override;
    void step() override;
    void compute_Scc() override;
    void _project_feasible_plain(Matf_X3& p, ProximalQuery::ContactInfoList& contacts, Real mu, size_t max_jacobi_iter) override;

    std::vector<std::array<float, 3>>& getContactPoints();
    std::vector<std::array<float, 3>>& getContactNormals();

    int Global_Jacobi_iter = 20;

    void do_compute_Scc() { compute_Scc(); }
    void do_project_feasible_parallel() { project_feasible_parallel(); }

protected:
    void project_feasible_parallel();
    void _project_feasible_impl();
    void compute_Scc_parallel();
    void compute_Scc_impl();
    void GS_global(const Matf_X3& b, Matf_X3& x_curr, int iter_cnt);
    void Jacobi_global(int iter_cnt);

private:
    std::unique_ptr<CompactSparseMat> comp_mat;
    std::unique_ptr<CuSolverData> solver_data_device;
    std::unique_ptr<CompactSparseMat> m_D_device;
    std::unique_ptr<CompactSparseMat> m_dt2DTWeTWe_device;
    std::unique_ptr<CompactSparseMat> m_M_device;
    std::unique_ptr<CompactSparseMat> m_dt2Wc_device;

    std::unique_ptr<TriangleConstraintDevice> triangle_constraint_cu;
    int triangle_constraint_start_row{};
    std::unique_ptr<PinConstraintDevice> pin_constraint_cu;
    int pin_constraint_start_row{};
    std::unique_ptr<BendingConstraintDevice> bending_constraint_cu;
    int bending_constraint_start_row{};
    std::unique_ptr<TetrahedralConstraintDevice> tet_constraint_cu;
    int tet_constraint_start_row{};

    thrust::device_vector<Real> DX_device;
    thrust::device_vector<Real> m_Ue_device;
    thrust::device_vector<Real> m_Uc_device;
    thrust::device_vector<Real> M_x_tilde_device;
    thrust::device_vector<Real> cache_nDynVertX3;
    thrust::device_vector<Real> cache_nDynVertX3_bp1;
    thrust::device_vector<Real> cache_nCDimX3;

    Matf_X3 jacobi_buffer;
    Matf_X3 x_curr;
    Matf_X3 p;
    Matf_X3 b_curr;

    std::vector<tbb::concurrent_vector<Real>> Gamma_i;
    std::vector<Vecf_4> Gamma_c_aux;
    std::vector<tbb::concurrent_vector<int>> involved_cid;
    std::vector<Vecf_3> delta_u;
    std::vector<int> vi_ct_nums;

    std::shared_ptr<ContactDataDevice> ct_data_device{};
    std::vector<std::array<float, 3>> contact_points;
    std::vector<std::array<float, 3>> contact_normals;

    thrust::device_vector<int> d_is_fixed;
    thrust::device_vector<Real> d_M;
};

} // namespace ADU
