#include "solver/solver_config_applier.h"

namespace ADU {

ADMMSolverConfig build_solver_config(const APPConfig& cfg, const SolverSetupContext& ctx) {
    ADMMSolverConfig c;

    // --- mesh / setup ---
    c.static_mesh_id_begin = ctx.static_mesh_id_begin;
    c.static_vert_begin = ctx.static_vert_begin;
    c.vinds_surf_set = ctx.surf_vinds_set;
    c.is_static = ctx.is_static;

    // --- integration / iterations ---
    c.admm_max_iter = cfg.solver.admm.admm_max_iter;
    c.DCD_interval = cfg.solver.admm.DCD_interval;
    c.gs_max_iter = cfg.solver.admm.GS_max_iter;
    c.global_jacobi_iter = cfg.solver.admm.Global_Jacobi_iter;
    c.warmstart_Ue = cfg.solver.admm.warmstart_Ue;
    c.warmstart_Uc = cfg.solver.admm.warmstart_Uc;
    c.use_jacobi = cfg.solver.contact.use_jacobi;
    c.g = cfg.global.g;
    c.dump_system_matrix = cfg.solver.admm.dump_A;
    c.coloring_parallel_contact = cfg.solver.contact.coloring;

    // --- frictional contact ---
    c.enable_frictional_contact = cfg.solver.contact.enable;
    c.use_CCD = cfg.solver.contact.use_CCD;
    c.use_unique_contact = cfg.solver.contact.unique;
    c.mu = cfg.solver.contact.mu;
    c.kappa = cfg.solver.contact.kappa;
    c.beta = cfg.solver.contact.beta;

    c.contact_w_list.resize(ctx.mesh->verts.rows());
    c.contact_w_inv_list.resize(ctx.mesh->verts.rows());
    c.contact_w_list.setConstant(1e12_r);
    c.contact_w_inv_list.setZero();
    if (cfg.solver.contact.use_heu_wc) {
        c.use_heuristic_wc = true;
        c.wc_sigma = cfg.solver.contact.wc_sigma;
        c.wc_beta = cfg.solver.contact.wc_beta;
    } else {
        for (int vi = 0; vi < ctx.static_vert_begin; vi++) {
            c.contact_w_list(vi) = ctx.mass(vi) / (ctx.dt * ctx.dt) * cfg.solver.contact.w_scale;
            c.contact_w_inv_list(vi) = 1.0_r / c.contact_w_list(vi);
        }
    }

    // --- damping ---
    c.damp_k_L = cfg.solver.damp.kl;
    c.damp_k_M = cfg.solver.damp.km;

    return c;
}

} // namespace ADU
