#pragma once

#include "mutils/common_types.h"

#include <unordered_set>

namespace ADU {

/// ADMM tuning parameters, derived from APPConfig by the application layer
/// (mediator/solver_config_applier.h) and handed to the backend through
/// ADMMSolver::apply_config(). Plain data struct: no solver/impl types, so
/// any backend (CPU/GPU/future) can consume it.
struct ADMMSolverConfig {
    // --- mesh / setup ---
    int static_mesh_id_begin{};
    int static_vert_begin{};
    ADU::Veci_X is_static;
    std::unordered_set<int> vinds_surf_set;

    // --- integration / iterations ---
    int admm_max_iter{ 25 };
    int DCD_interval{ 5 };
    int gs_max_iter{ 10 };
    int global_jacobi_iter{ 20 };
    bool warmstart_Ue{ true };
    bool warmstart_Uc{ true };
    bool use_jacobi{ false };
    ADU::Real g{ static_cast<ADU::Real>(0.98) };

    // --- frictional contact ---
    bool enable_frictional_contact{ true };
    bool use_CCD{ false };
    bool use_unique_contact{ false };
    ADU::Real mu{ static_cast<ADU::Real>(0.5) };
    ADU::Real kappa{ static_cast<ADU::Real>(1e6) };
    ADU::Real beta{ static_cast<ADU::Real>(1e5) };
    bool use_heuristic_wc{ false };
    ADU::Real wc_sigma{ static_cast<ADU::Real>(0.001) };
    ADU::Real wc_beta{ static_cast<ADU::Real>(25) };
    ADU::Vecf_X contact_w_list;
    ADU::Vecf_X contact_w_inv_list;

    // --- damping ---
    ADU::Real damp_k_L{ static_cast<ADU::Real>(1e-4) };
    ADU::Real damp_k_M{ static_cast<ADU::Real>(0.0) };
};

} // namespace ADU
