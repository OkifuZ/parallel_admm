#include <filesystem>
#include <iostream>

#include <Eigen/Core>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "mediator/app_context.h"
#include "mediator/app_initializer.h"
#include "mediator/bvh_display_helper.h"
#include "mediator/contact_display_helper.h"
#include "mediator/export_frame.h"
#include "solver/core/admm_solver.h"
#include "mutils/timer.h"
#include "src_config.h"

using namespace ADU;

namespace {
constexpr int kEigenNumThreads = 12;
}

AppContext g_ctx;

void main_loop() {
    ImGui::CheckboxFlags("pause", (unsigned int*)&g_ctx.app.pause, ImGuiConfigFlags_NavEnableKeyboard); ImGui::SameLine();
    if (ImGui::Button("step")) g_ctx.app.step = true;
    if (ImGui::Button("reset")) g_ctx.app.reset = true;
    ImGui::CheckboxFlags("export obj", (unsigned int*)&g_ctx.app.export_obj, ImGuiConfigFlags_NavEnableKeyboard);

    if (g_ctx.app.reset) {
        g_ctx.app.solver->reset(g_ctx.app.mesh->verts, false);
        g_ctx.app.mesh->clear_color();
        g_ctx.app.reset = false;
        g_ctx.app.pause = true;
        g_ctx.step_idx = 0;
    }

    if (!g_ctx.app.pause || g_ctx.app.step) {
        std::cout << "step frame " << g_ctx.step_idx << "\n";

        g_ctx.app.step = false;

        if (g_ctx.app.enable_XPBD && g_ctx.app.XPBD_solver) {
            g_ctx.app.XPBD_solver->step();
        } else {
            for (int i = 0; i < g_ctx.app.sub_step; i++) g_ctx.app.solver->step();
        }

        if (g_ctx.app.show_windows && g_ctx.app.bvh) {
            update_bvh_geometry(g_ctx.app.bvh, g_ctx.app.bvh_nodes, g_ctx.app.bvh_edges);
        }

        if (g_ctx.app.export_obj && g_ctx.app.save_res) {
            export_frame(g_ctx.out_path, g_ctx.config.scene_name, g_ctx.step_idx,
                g_ctx.app.solver->getVertices(), g_ctx.app.mesh->surface_tris,
                g_ctx.config.separate_out ? g_ctx.app.mesh.get() : nullptr,
                g_ctx.app.use_bin, g_ctx.config.separate_out);
        }

        if (g_ctx.app.show_windows && g_ctx.app.vis_meshP) {
            g_ctx.app.vis_meshP->addVertexVectorQuantity("velocity", g_ctx.app.solver->getVelocities());
        }

        if (g_ctx.app.show_windows && g_ctx.app.end_of_step_callback) {
            g_ctx.app.end_of_step_callback();
        }

        g_ctx.step_idx += 1;
    }

    if (g_ctx.app.show_windows && g_ctx.app.vis_meshP) {
        g_ctx.app.vis_meshP->updateVertexPositions(g_ctx.app.solver->getVertices());
        g_ctx.app.mesh->clear_color();
    }
    if (g_ctx.app.show_windows && g_ctx.app.vis_contactP) {
        auto* admm_solver = static_cast<ADMMSolver*>(g_ctx.app.solver.get());
        update_contact_display(admm_solver, g_ctx.config.use_GPU, g_ctx.app.vis_contactP);
    }
    if (g_ctx.app.show_windows && g_ctx.app.vis_bvh) {
        g_ctx.app.vis_bvh->updateNodePositions(g_ctx.app.bvh_nodes);
    }

    if (g_ctx.step_idx > g_ctx.app.end_frame) {
        g_ctx.app.pause = true;
        polyscope::unshow();
    }
}

int main(int argc, const char* argv[]) {
#ifdef ADMM_DEBUG
    AppInitParams params;
    params.config_path = (std::filesystem::path(PROJECT_RESOURCE_PATH) / "scene/test_XPBD_ADMM.toml").string();
    params.resource_path = std::filesystem::path(PROJECT_RESOURCE_PATH);
#else
    AppInitParams params;
    try {
        params = parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
#endif

    Eigen::setNbThreads(kEigenNumThreads);
    Eigen::initParallel();

    init_app(g_ctx, params);

    // Initial BVH geometry for polyscope registration
    update_bvh_geometry(g_ctx.app.bvh, g_ctx.app.bvh_nodes, g_ctx.app.bvh_edges);

    polyscope::init();

    if (g_ctx.app.show_windows) {
        Solver* inner = static_cast<ADMMSolver*>(g_ctx.app.solver.get())->inner_solver();

        polyscope::registerSurfaceMesh("surface", g_ctx.app.mesh->verts, g_ctx.app.mesh->surface_tris);
        polyscope::registerPointCloud("contact", inner->prox_query->get_contact_points());
        polyscope::registerCurveNetwork("bvh", g_ctx.app.bvh_nodes, g_ctx.app.bvh_edges);

        g_ctx.app.vis_bvh = polyscope::getCurveNetwork("bvh");
        g_ctx.app.vis_bvh->setEnabled(false);

        if (!g_ctx.config.global.pin_ids.empty()) {
            polyscope::registerPointCloud("pin", g_ctx.app.pin_v);
            g_ctx.app.pin_pts = polyscope::getPointCloud("pin");
        }

        polyscope::view::resetCameraToHomeView();
        polyscope::options::automaticallyComputeSceneExtents = false;
        polyscope::state::lengthScale = g_ctx.config.viewer.scale;

        g_ctx.app.vis_meshP = polyscope::getSurfaceMesh("surface");
        g_ctx.app.vis_contactP = polyscope::getPointCloud("contact");
        g_ctx.app.vis_contactP->setEnabled(false);
        g_ctx.app.vis_meshP->addFaceColorQuantity("mesh color", g_ctx.app.mesh->f_colors);

        polyscope::options::groundPlaneEnabled = g_ctx.config.viewer.enable_ground;
        polyscope::view::bgColor = {
            (float)g_ctx.config.viewer.background_color.x(),
            (float)g_ctx.config.viewer.background_color.y(),
            (float)g_ctx.config.viewer.background_color.z(),
            (float)0.0_r};
    }

    polyscope::state::userCallback = main_loop;
    polyscope::show();

    if (g_ctx.config.use_GPU) {
        std::cout << "max collision num: " << g_ctx.app.bvh->hist_max_collision_num << "\n";
    } else {
        Solver* inner = static_cast<ADMMSolver*>(g_ctx.app.solver.get())->inner_solver();
        std::cout << "max collision num: " << inner->prox_query->hist_max_collision_num << "\n";
    }
    std::cout << "logger:\n";
    std::cout << ADU::Timer::getLog();
}
