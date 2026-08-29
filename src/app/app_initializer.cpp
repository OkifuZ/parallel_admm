#include "app/app_initializer.h"
#include "app/app_context.h"
#include "contact/dcd_validation.h"
#include "config/mesh_from_config.h"
#include "constraint/mesh_to_constraint.h"
#include "constraint/xpbd/mesh_to_xpbd_constraint.h"
#include "solver/solver_config_applier.h"
#include "config/app_config.h"
#include "solver/admm_solver_factory.h"
#include "solver/core/admm_solver.h"
#include "solver/xpbd_solver.h"
#include "contact/broad_phase.h"
#include "contact/narrow_phase.h"
#include "mesh/mesh_container.h"
#include "mutils/aux_color.h"

#include <argparse/argparse.hpp>
#include <Eigen/Core>
#include <iostream>
#include <unordered_set>

namespace ADU {

AppInitParams parse_args(int argc, const char* argv[]) {
    argparse::ArgumentParser program("admm_elasticity");
    program.add_argument("config_file_path").help("path of scene config file");
    program.add_argument("resource_file_path").help("path of dir of resource");
    program.parse_args(argc, argv);

    AppInitParams params;
    params.config_path = program.get<std::string>("config_file_path");
    params.resource_path = std::filesystem::path(program.get<std::string>("resource_file_path"));
    return params;
}

void init_app(AppContext& ctx, const AppInitParams& params) {
    const auto& config_path = params.config_path;
    const auto& resource_path = params.resource_path;

    ctx.config.load_config(config_path);

    ctx.app.name = ctx.config.scene_name;
    ctx.out_path = std::filesystem::path(ctx.config.out_file);

    if (!std::filesystem::exists(ctx.out_path)) {
        try {
            std::filesystem::create_directories(ctx.out_path);
            ctx.app.save_res = true;
        } catch (const std::exception&) {
            ctx.app.save_res = false;
        }
    } else {
        ctx.app.save_res = true;
    }

    ctx.app.use_bin = ctx.config.out_bin;
    ctx.app.export_obj = ctx.config.export_obj;
    ctx.app.show_windows = ctx.config.show_windows;
    ctx.app.end_frame = ctx.config.end_frame;
    ctx.app.sub_step = ctx.config.global.sub_step;
    ctx.app.dt = ctx.config.global.dt / ctx.app.sub_step;

    ctx.app.mesh = std::make_shared<MeshData>();

    std::cout << "loading mesh...\n";
    auto load_result = load_meshes_from_config(*ctx.app.mesh, ctx.config, resource_path);
    auto& mesh_id_list = load_result.mesh_id_list;
    auto& material_list = load_result.material_list;
    int static_mesh_id_begin = load_result.static_mesh_id_begin;
    int static_vert_begin = load_result.static_vert_begin;
    auto& mesh_animate_info = load_result.mesh_animate_info;
    std::cout << "done loading mesh\n";

    ctx.app.mesh->precompute_geometry_info();
    ctx.app.mesh->static_mesh_id_begin = static_mesh_id_begin;
    ctx.app.mesh->static_vert_begin = static_vert_begin;

    std::unordered_set<int> surf_vinds_set;
    for (int i = 0; i < ctx.app.mesh->surface_vinds.rows(); i++) {
        surf_vinds_set.insert(ctx.app.mesh->surface_vinds(i));
    }

    Veci_X is_static(ctx.app.mesh->verts.rows());
    for (auto& m : ctx.app.mesh->mesh_list) {
        for (int i = m->start_vertIdx; i < m->end_vertIdx; i++) {
            is_static(i) = int(m->is_static);
        }
    }
    ctx.app.mesh->is_static = is_static;

    int mi = 0;
    for (auto& m : ctx.app.mesh->mesh_list) {
        std::cout << "mesh " << mi++ << ": " << m->start_vertIdx << "-" << (m->end_vertIdx - 1) << "\n";
        auto ccc = CoolColor::color(m->color_id);
        std::cout << "(" << ccc.x() << ", " << ccc.y() << ", " << ccc.z() << ")\n";
    }
    std::cout << "tet number: " << ctx.app.mesh->tets.rows() << "\n";

    ctx.app.mesh->get_mass(ctx.app.mass);

    // --- solver creation & typed configuration ---
    // Selection: explicit [solver] type, or legacy [xpbd] enable=true.
    SolverType solver_type = SolverType::ADMM_CPU;
    if (ctx.config.solver.type == "xpbd" || ctx.config.xpbd.use_XPBD) {
        solver_type = SolverType::XPBD;
    } else {
        solver_type = ctx.config.use_GPU ? SolverType::ADMM_GPU : SolverType::ADMM_CPU;
    }
    ctx.app.solver = create_solver(solver_type);
    ADMMSolver* admm_solver = nullptr;
    Solver* inner = nullptr;

    if (solver_type == SolverType::XPBD) {
        auto* xpbd = static_cast<XPBDSolver*>(ctx.app.solver.get());
        inner = xpbd;

        XPBDConfig xpbd_cfg;
        xpbd_cfg.substeps = ctx.config.xpbd.substeps;
        xpbd_cfg.max_iter = ctx.config.xpbd.XPBD_iter;
        xpbd_cfg.contact_stiffness = ctx.config.xpbd.contact_stiffness;
        xpbd_cfg.use_GS_contact = ctx.config.xpbd.use_GS_contact;
        xpbd_cfg.g = ctx.config.global.g;
        xpbd_cfg.enable_frictional_contact = ctx.config.solver.contact.enable;
        xpbd_cfg.dcd_interval = ctx.config.solver.admm.DCD_interval;
        xpbd_cfg.use_unique_contact = ctx.config.solver.contact.unique;
        xpbd_cfg.mu = ctx.config.solver.contact.mu;
        xpbd->apply_config(xpbd_cfg);

        // XPBD constraints: tri stretch / tet strain / bending (per mesh).
        XPBDSolver::XPBDConstraintList xpbd_cslist;
        for (size_t i = 0; i < mesh_id_list.size(); i++) {
            build_xpbd_constraints(*ctx.app.mesh, mesh_id_list[i], material_list[i], xpbd_cslist);
        }
        std::cout << "[xpbd] constraints built: " << xpbd_cslist.size() << std::endl;
        xpbd->set_constraints(xpbd_cslist);
    } else {
        admm_solver = static_cast<ADMMSolver*>(ctx.app.solver.get());
        inner = admm_solver->inner_solver();

        SolverSetupContext solver_ctx{ ctx.app.mass, ctx.app.dt, is_static,
            static_mesh_id_begin, static_vert_begin, surf_vinds_set, ctx.app.mesh.get() };
        admm_solver->apply_config(build_solver_config(ctx.config, solver_ctx));

        Solver::ConstraintsList cslist;
        for (int tid = 1; tid <= 3; tid++) {
            for (size_t i = 0; i < mesh_id_list.size(); i++) {
                Mesh2Constraint::geometry_to_constraints(*ctx.app.mesh, mesh_id_list[i], material_list[i], cslist, tid);
            }
        }

        // Constraint dimension is known only after assembly; the pin
        // constraints appended below also occupy rows, so set the dimension
        // and finalize AFTER pin_to_constraints (regression fix: the GPU
        // convert_constraint2device also needs the complete list).
        admm_solver->add_constraints(cslist);
    }
    ctx.app.inner = inner;

    // Generic setup shared by all solver types.
    inner->addPins(ctx.config.global.pin_ids);

    ctx.app.pin_v.resize(ctx.config.global.pin_ids.size(), 3);
    for (int pi = 0; pi < ctx.config.global.pin_ids.size(); pi++) {
        ctx.app.pin_v.row(pi) = ctx.app.mesh->verts.row(ctx.config.global.pin_ids[pi]);
    }
    if (solver_type != SolverType::XPBD) {
        // ADMM pin constraints (positions driven through the prox step).
        Solver::ConstraintsList& cslist = admm_solver->inner_solver()->getConstraints();
        Mesh2Constraint::pin_to_constraints(*ctx.app.mesh, ctx.config.global.pin_ids, cslist,
            ctx.app.pin_v, is_static, ctx.app.pin_start, ctx.app.pin_end);
        admm_solver->set_constraint_dim(Mesh2Constraint::curr_start_row);
        admm_solver->finalize_constraints();
    }

    ContactParameter::set_broadphase_radius(ctx.config.dcd.broad.radius);
    ContactParameter::set_thickness(ctx.config.dcd.narrow.thickness);

    inner->prox_query = std::make_unique<ProximalQuery>(ctx.app.mesh, ctx.config.dcd.narrow.max_collision);
    BroadPhase_embree::BVH_TYPE bvh_type =
        (ctx.config.dcd.broad.strategy == "refit") ? BroadPhase_embree::BVH_TYPE::REFIT : BroadPhase_embree::BVH_TYPE::REBUILD;

    inner->prox_query->broad_phase = std::make_shared<BroadPhase_embree>(bvh_type);
    inner->prox_query->broad_phase->mesh = ctx.app.mesh;
    inner->prox_query->broad_phase->build();

    if (!thickness_validation_check(*(ctx.app.mesh))) {
        throw std::runtime_error("thickness_validation_check failed");
    }

    for (size_t i = 0; i < ctx.config.colliders.size(); i++) {
        inner->addObstacle(
            std::make_shared<PlaneObstacle>(
                ctx.config.colliders[i].center,
                ctx.config.colliders[i].halfside,
                ctx.config.colliders[i].normal));
    }

    if (!mesh_animate_info.empty()) {
        inner->animator = std::make_unique<ScriptAnimator>();
        inner->animator->mesh = ctx.app.mesh;
        for (const auto& p : mesh_animate_info) {
            inner->animator->attach_animate2mesh(p.first, (resource_path / p.second).string());
        }
    }

    ctx.app.solver->init(ctx.app.mesh->verts, ctx.app.mass, ctx.app.dt);

    // Headless runs (no window, no export, no animator) never read host x/v,
    // so the GPU backend can skip the per-step device -> host download.
    if (admm_solver) {
        admm_solver->set_sync_to_host(ctx.app.show_windows || (inner->animator != nullptr));
    }

    ctx.app.bvh_holder = std::make_shared<BVH_GPU>();
    ctx.app.bvh = ctx.app.bvh_holder.get();
    BVH_GPU& bvh = *ctx.app.bvh_holder;
    bvh.solver = inner;
    // NOTE: set the BVH on the INNER solver (which owns the simulation state);
    // the facade's inherited member is never used.
    inner->bvh = &bvh;
    bvh.mesh = ctx.app.mesh.get();

    ctx.app.bvh->init(inner->prox_query->max_collision_num);
    ctx.app.bvh->construct(true);
}

} // namespace ADU

