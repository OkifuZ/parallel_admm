#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/vec3.hpp>

#include "mutils/common_types.h"
#include "contact/collision_detection_cuda.h"
#include "mesh/mesh_container.h"
#include "solver/solver.h"
#include "config/app_config.h"

namespace polyscope {
class SurfaceMesh;
class PointCloud;
class CurveNetwork;
}

namespace ADU {

/// Application state: solver, mesh, visualization handles.
struct APP {
    std::string name{};
    std::unique_ptr<Solver> solver;

    int end_frame = 2000;

    std::shared_ptr<MeshData> mesh;
    Vecf_X mass;
    Real dt = 0.0333_r;
    int sub_step = 1;

    int pin_start{};
    int pin_end{};
    Matf_XX pin_v;

    polyscope::SurfaceMesh* vis_meshP{};
    polyscope::PointCloud* vis_contactP{};
    polyscope::PointCloud* pin_pts{};
    polyscope::CurveNetwork* vis_bvh{};

    unsigned int pause = 1;
    unsigned int export_obj = 0;
    bool use_bin = false;
    bool step = false;
    bool reset = false;
    bool save_res = false;
    bool show_windows = true;

    std::function<void()> end_of_step_callback;

    std::vector<glm::vec3> bvh_nodes;
    std::vector<std::array<size_t, 2>> bvh_edges;

    std::shared_ptr<BVH_GPU> bvh_holder;
    BVH_GPU* bvh{};
};

/// Consolidated app state.
struct AppContext {
    APP app;
    APPConfig config;
    std::filesystem::path out_path;
    size_t step_idx = 0;
};

} // namespace ADU
