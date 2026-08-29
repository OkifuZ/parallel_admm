#pragma once

#include "mediator/toml_to_config.h"
#include "mesh/mesh_container.h"
#include "mutils/common_types.h"
#include "solver/solver.h"

namespace ADU {

/// Runtime context needed to apply solver config (mesh-derived data).
struct SolverSetupContext {
    const Vecf_X& mass;
    Real dt;
    const Veci_X& is_static;
    int static_mesh_id_begin;
    int static_vert_begin;
    std::unordered_set<int> surf_vinds_set;
    const MeshData* mesh;
};

/// Apply APPConfig to inner solver. Accepts Solver*; casts internally.
/// Application layer does not need impl headers.
void apply_solver_config(Solver* inner, const APPConfig& cfg, const SolverSetupContext& ctx);

/// Setup GPU-specific config and convert_constraint2device. No-op when inner is CPU impl.
void setup_parallel_backend(Solver* inner, const APPConfig& cfg);

} // namespace ADU
