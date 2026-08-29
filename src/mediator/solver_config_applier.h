#pragma once

#include "mediator/toml_to_config.h"
#include "mesh/mesh_container.h"
#include "mutils/common_types.h"
#include "solver/core/admm_config.h"

namespace ADU {

/// Runtime context needed to derive solver config (mesh-derived data).
struct SolverSetupContext {
    const Vecf_X& mass;
    Real dt;
    const Veci_X& is_static;
    int static_mesh_id_begin;
    int static_vert_begin;
    std::unordered_set<int> surf_vinds_set;
    const MeshData* mesh;
};

/// Derive ADMM tuning parameters from the scene config.
/// Pure data: no solver/impl types involved, so any backend can consume it.
/// (Replaces the old apply_solver_config/setup_parallel_backend dynamic_cast path.)
ADMMSolverConfig build_solver_config(const APPConfig& cfg, const SolverSetupContext& ctx);

} // namespace ADU
