#pragma once

/// Phase 3.4: CPU ADMM implementation (migrated from admm_full_solver.h).
/// Logic lives here; ADMMBackendCPU holds this via composition.

#include "mutils/common_types.h"
#include <mutils/common_type_hostonly.h>

#include "solver/compact_sparse_matrix.h"
#include "solver/solver.h"
#include "constraint/constraint.h"
#include "solver/animator.h"
#include "contact/narrow_phase.h"
#include "solver/jacobi_solver.h"

#include <tbb/concurrent_vector.h>
#include <random>

#include "constraint/pin_constraint_device.h"
#include "constraint/triangle_constraint_device.h"
#include "constraint/bending_constraint_device.cuh"

namespace ADU {

class ADMMImplCPUBase : public Solver {
public:
    bool use_jacobi = false;
    size_t max_remaining = 0;
    int _color_max_num = 16;
    std::random_device rd;

    using TripList = std::vector<ADU::Tripf>;
    using SolverT = Eigen::SimplicialLLT<ADU::SpMatf>;

    bool enable_frictional_contact{ true };
    bool warmstart_Ue{ true };
    bool warmstart_Uc{ true };
    size_t collision_detection_interval = 5;
    size_t gs_max_iter = 10;
    ADU::Vecf_X contact_w_list;
    ADU::Vecf_X contact_w_inv_list;
    ADU::Real mu{ static_cast<ADU::Real>(0.5) };
    ADU::Real damping_ratio{ static_cast<ADU::Real>(0.99) };
    bool coloring_parallel_contact{ false };
    bool color_use_random{ false };
    ADU::Real kappa = static_cast<ADU::Real>(1e6);
    ADU::Real beta = static_cast<ADU::Real>(1e5);

    ADMMImplCPUBase() : Solver() {}

    virtual void precompute();
    virtual void init();
    virtual void reset(const ADU::Matf_X3& ini_verts, bool need_precompute = false);
    virtual void step() = 0;
    virtual void project_feasible(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_GS_iter);
    void sequential_greedy_coloring(const ProximalQuery::ContactInfoList& cs_list, bool use_rand = false);

    std::vector<int> ct_colors;
    ADU::SpMatf m_A;
    ADU::SpMatf m_M;
    ADU::SpMatf m_M_flat;
    std::vector<std::vector<size_t>> v_c_list;
    std::vector<std::vector<size_t>> color_result;
    ADU::SpMatf m_D;
    ADU::SpMatf m_W_e;
    ADU::SpMatf m_dt2DTWeTWeD;
    ADU::SpMatf m_dt2DTWeTWe;
    ADU::SpMatf m_W_c;
    ADU::SpMatf m_W_c_flat;
    ADU::SpMatf m_dt2Wc;
    ADU::Matf_X3 m_Ue;
    ADU::Matf_X3 m_Uc;
    bool need_recompute_Scc{ false };
    std::vector<ADU::Vecf_4> Gamma_c;
    std::vector<ADU::Vecf_3> K_c;
    ADU::Real epsilon{};
    ADU::Real gamma{};
    std::unique_ptr<SolverT> m_LLT_solver;
};

/// RL-damping CPU implementation (migrated from ADMMSolverFull_RL_damping).
class ADMMImplCPU : public ADMMImplCPUBase {
public:
    bool enable_large_scale{ false };
    bool use_heuristic_W_c{ false };
    ADU::Real heu_sigma{ 0.001 };
    ADU::Real heu_beta{ 25 };
    int static_mesh_id_begin{};
    int static_vert_begin{};
    bool out_residual = false;

    ADU::SpMatf m_Damp_Mat;
    ADU::SpMatf m_A_damp;
    ADU::SpMatf m_I;
    ADU::Real damp_k_L{ ADU::toReal(1e-4) };
    ADU::Real damp_k_M{ ADU::toReal(0.0) };
    bool use_anisotropy = false;
    ADU::Real mu_t = 0.05;
    ADU::Real mu_b = 1.8;
    ADU::Matf_X3 tangent_vec;
    bool use_CCD{ false };
    bool use_unique_contact{ false };

    virtual void init();
    virtual void precompute();
    virtual void project_feasible(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_jacobi_iter);
    virtual void _project_feasible_plain(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_jacobi_iter);

    std::unordered_set<int> vinds_surf_set;
    ADU::Matf_X3 x_prev;
    std::vector<std::vector<ADU::Real>> rr_list;
    bool use_NNCG{ false };
    ADU::Matf_X3 x_km1_c, g_k_c, g_km1_c, p_cg_c;
    ADU::Real g_k_sqn_c, g_km1_sqn_c;
    ADU::Matf_X3 x_km1_e, g_k_e, g_km1_e, p_cg_e;
    ADU::Real g_k_sqn_e, g_km1_sqn_e;

    std::vector<ProximalQuery::ContactInfoList> contact_islands;
    void find_contact_islands();

    virtual void compute_Scc(bool is_XPBD = false);
    virtual void step();
    void step_fast();

    int m_maxXPBDIterations = 50;
    int step_cnt = 0;
    std::vector<std::vector<ADU::Real>> primal_residual, dual_residual, combined_residual, x_residual, time_spans;
    void compute_SaSb();

    int mesh_num{ 0 };
    std::vector<std::pair<int, int>> se_idx_list;
    std::vector<ADU::SpMatf> m_sub_A_list;
    std::vector<std::unique_ptr<SolverT>> m_LLT_solver_list;
};

} // namespace ADU
