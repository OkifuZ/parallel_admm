#include <filesystem>
#include <iostream>

#include <Eigen/Core>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "app/app_context.h"
#include "app/app_initializer.h"
#include "contact/bvh_display.h"
#include "contact/contact_display.h"
#include "mesh/export_frame.h"
#include "solver/core/admm_solver.h"
#include "contact/icollision_detector.h"
#include "mutils/timer.h"
#include "src_config.h"

using namespace ADU;

namespace {
constexpr int kEigenNumThreads = 12;
}

AppContext g_ctx;

void main_loop() {
    // Headless runs (show_windows = false) must not wait for GUI input:
    // auto-run until end_frame, then unshow and exit.
    if (!g_ctx.app.show_windows) g_ctx.app.pause = false;

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

        for (int i = 0; i < g_ctx.app.sub_step; i++) g_ctx.app.solver->step();

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
        ICollisionDetector* det = g_ctx.config.use_GPU
            ? static_cast<ICollisionDetector*>(g_ctx.app.bvh)
            : static_cast<ICollisionDetector*>(g_ctx.app.inner->prox_query.get());
        update_contact_display(det, g_ctx.app.inner, g_ctx.app.vis_contactP);
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
        Solver* inner = g_ctx.app.inner;

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

    Solver* inner = g_ctx.app.inner;
    ICollisionDetector* det = g_ctx.config.use_GPU
        ? static_cast<ICollisionDetector*>(g_ctx.app.bvh)
        : static_cast<ICollisionDetector*>(inner->prox_query.get());
    std::cout << "max collision num: " << det->peak_contact_count() << "\n";

    // Per-step metrics CSV (frame, step_ms, n_contacts) for solver comparison.
    if (g_ctx.app.save_res && !inner->step_metrics.empty()) {
        const auto metrics_path = g_ctx.out_path / "step_metrics.csv";
        inner->write_step_metrics_csv(metrics_path.string());
        std::cout << "step metrics: " << metrics_path << "\n";
    }

    std::cout << "logger:\n";
    std::cout << ADU::Timer::getLog();
}
