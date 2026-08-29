#include "mediator/app_initializer.h"
#include "mediator/app_context.h"
#include "mediator/DCD_validation_check.h"
#include "mediator/mesh_from_config.h"
#include "mediator/mesh_to_constraint.h"
#include "mediator/solver_config_applier.h"
#include "mediator/toml_to_config.h"
#include "solver/admm_solver_factory.h"
#include "solver/core/admm_solver.h"
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

    ctx.app.solver = create_solver(ctx.config.use_GPU ? SolverType::ADMM_GPU : SolverType::ADMM_CPU);
    auto* admm_solver = static_cast<ADMMSolver*>(ctx.app.solver.get());
    Solver* inner = admm_solver->inner_solver();

    SolverSetupContext solver_ctx{ ctx.app.mass, ctx.app.dt, is_static,
        static_mesh_id_begin, static_vert_begin, surf_vinds_set, ctx.app.mesh.get() };
    admm_solver->apply_config(build_solver_config(ctx.config, solver_ctx));

    Solver::ConstraintsList cslist;
    for (int tid = 1; tid <= 3; tid++) {
        for (size_t i = 0; i < mesh_id_list.size(); i++) {
            Mesh2Constraint::geometry_to_constraints(*ctx.app.mesh, mesh_id_list[i], material_list[i], cslist, tid);
        }
    }

    admm_solver->addPins(ctx.config.global.pin_ids);

    ctx.app.pin_v.resize(ctx.config.global.pin_ids.size(), 3);
    Mesh2Constraint::pin_to_constraints(*ctx.app.mesh, ctx.config.global.pin_ids, cslist, ctx.app.pin_v, is_static, ctx.app.pin_start, ctx.app.pin_end);

    admm_solver->add_constraints(cslist);

    // Constraint dimension is known only after assembly; finalize AFTER all
    // constraints are registered so the GPU backend converts a complete list
    // (regression fix: convert_constraint2device used to run too early).
    admm_solver->set_constraint_dim(Mesh2Constraint::curr_start_row);
    admm_solver->finalize_constraints();

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
    admm_solver->set_sync_to_host(ctx.app.show_windows || (inner->animator != nullptr));

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
