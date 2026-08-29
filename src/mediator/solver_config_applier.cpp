#include "mediator/solver_config_applier.h"
#include "solver/backend_cpu/admm_impl_cpu.h"
#include "solver/backend_gpu/admm_impl_gpu.h"

namespace ADU {

void apply_solver_config(Solver* inner, const APPConfig& cfg, const SolverSetupContext& ctx) {
    auto* impl = dynamic_cast<ADMMImplCPU*>(inner);
    if (!impl) return;
    impl->static_mesh_id_begin = ctx.static_mesh_id_begin;
    impl->static_vert_begin = ctx.static_vert_begin;
    impl->vinds_surf_set = ctx.surf_vinds_set;
    impl->enable_frictional_contact = cfg.solver.contact.enable;
    impl->warmstart_Ue = cfg.solver.admm.warmstart_Ue;
    impl->warmstart_Uc = cfg.solver.admm.warmstart_Uc;
    impl->admm_max_iter = cfg.solver.admm.admm_max_iter;
    impl->collision_detection_interval = cfg.solver.admm.DCD_interval;
    impl->gs_max_iter = cfg.solver.admm.GS_max_iter;
    Real w_scale = cfg.solver.contact.w_scale;
    impl->kappa = cfg.solver.contact.kappa;
    impl->beta = cfg.solver.contact.beta;
    impl->mu = cfg.solver.contact.mu;
    impl->g = cfg.global.g;
    impl->use_CCD = cfg.solver.contact.use_CCD;
    impl->use_jacobi = cfg.solver.contact.use_jacobi;
    impl->coloring_parallel_contact = false;
    impl->damp_k_L = cfg.solver.damp.kl;
    impl->damp_k_M = cfg.solver.damp.km;
    impl->m_vStatic = ctx.is_static;
    impl->use_unique_contact = cfg.solver.contact.unique;
    impl->contact_w_list.resize(ctx.mesh->verts.rows());
    impl->contact_w_inv_list.resize(ctx.mesh->verts.rows());
    impl->contact_w_list.setConstant(1e12_r);
    impl->contact_w_inv_list.setZero();
    if (cfg.solver.contact.use_heu_wc) {
        impl->use_heuristic_W_c = true;
        impl->heu_sigma = cfg.solver.contact.wc_sigma;
        impl->heu_beta = cfg.solver.contact.wc_beta;
    } else {
        for (int vi = 0; vi < ctx.static_vert_begin; vi++) {
            impl->contact_w_list(vi) = ctx.mass(vi) / (ctx.dt * ctx.dt) * w_scale;
            impl->contact_w_inv_list(vi) = 1.0_r / impl->contact_w_list(vi);
        }
    }
}

void setup_parallel_backend(Solver* inner, const APPConfig& cfg) {
    auto* gpu = dynamic_cast<ADMMImplGPU*>(inner);
    if (!gpu) return;
    gpu->Global_Jacobi_iter = cfg.solver.admm.Global_Jacobi_iter;
    gpu->convert_constraint2device();
}

} // namespace ADU
