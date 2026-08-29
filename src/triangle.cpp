#include <igl/readOBJ.h>
#include <igl/boundary_facets.h>
#include <igl/adjacency_list.h>
#include <igl/unique.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "src_config.h"
#include "mesh/mesh_container.h"
#include "solver/backend_cpu/admm_impl_cpu.h"
#include "constraint/constraint.h"
#include "constraint/triangle_constraint.h"
#include "constraint/bending_constraint.h"
#include "constraint/pin_constraint.h"

#include "mutils/common_types.h"
#include "mutils/timer.h"

#include <string>
#include <filesystem>
#include <memory>
#include <unordered_set>

using namespace ADU;

class APP {
public:
    std::unique_ptr<Solver> solver;
    std::unique_ptr<MeshData> mesh;
    Vecf_X mass;
    Real dt = 0.0333_r;
    
    polyscope::SurfaceMesh* vis_meshP{};

    unsigned int pause = 1;
    bool step = false;
};

APP app;


void main_loop() {
    ImGui::CheckboxFlags("pause", (unsigned int*)&app.pause, ImGuiConfigFlags_NavEnableKeyboard);
    if (ImGui::Button("step")) app.step = true;

    if (!app.pause || app.step) {
        app.solver->step();
        app.step = false;
    }

    if (app.vis_meshP) {
        app.vis_meshP->updateVertexPositions(app.solver->getVertices());
    }
}


int sphere_count = 0;
int plane_count = 0;
std::filesystem::path resource_path(PROJECT_RESOURCE_PATH);
Matf_X3 sphere_vert;
Mati_XX sphere_face;
Matf_X3 plane_vert;
Mati_XX plane_face;
void register_obstacle_mesh(const std::shared_ptr<Obstacle>& obstacle) {

    using namespace ADU;
    
    if (obstacle->type == 0) { // sphere
        auto& sphere = std::static_pointer_cast<SphereObstacle>(obstacle);
        Real radius = sphere->radius * 0.99;
        Vecf_3 center = sphere->center;
        std::string name = "sphere collider " + std::to_string(++sphere_count);
        polyscope::registerSurfaceMesh(name, sphere_vert, sphere_face);
        auto mesh = polyscope::getSurfaceMesh(name);
        glm::mat4x4 transform_mat{
            radius, 0, 0, center.x(),
            0, radius, 0, center.y(),
            0, 0, radius, center.z(),
            0, 0, 0, 1
        };
        transform_mat = glm::transpose(transform_mat);
        mesh->setTransform(transform_mat);
    }
    else if (obstacle->type == 1) { // plane
        auto& plane = std::static_pointer_cast<PlaneObstacle>(obstacle);
        Real scale = plane->half_edge;
        Vecf_3 center = plane->center;
        center.y() -= 0.01_r;
        std::string name = "plane collider " + std::to_string(++plane_count);
        polyscope::registerSurfaceMesh(name, plane_vert, plane_face);
        auto mesh = polyscope::getSurfaceMesh(name);
        glm::mat4x4 transform_mat{
            scale, 0, 0, center.x(),
            0, scale, 0, center.y(),
            0, 0, scale, center.z(),
            0, 0, 0, 1
        };
        transform_mat = glm::transpose(transform_mat);
        mesh->setTransform(transform_mat);
    }
    else {}
}


int main() {
    // static mesh
    igl::readOBJ((resource_path / "trimesh/unit_ball.obj").string(), sphere_vert, sphere_face);
    igl::readOBJ((resource_path / "flat/flat.obj").string(), plane_vert, plane_face);

    // simulated mesh
    app.mesh = std::make_unique<MeshData>();
    int mesh_id = app.mesh->add_mesh((resource_path / "trimesh/bowl.obj").string(), MeshData::TRI);
    //app.mesh = std::make_unique<MeshData>((resource_path / "flat/flat_v3452.obj").string(), MeshData::TRI);
    //app.mesh->scale(7.0_r);
    //app.mesh->translate({ 0, 4.2_r, 0 });
    app.mesh->translate({ 0, 2.5_r, 0 }, mesh_id);
    
    app.mesh->precompute_geometry_info();

    app.mesh->get_mass(1.0_r, app.mass);

    app.solver = std::make_unique<ADMMImplCPU>();

    Solver::ConstraintsList cslist;
    int start_row = 0;

    // stretching
    for (int fi = 0; fi < app.mesh->faces.rows(); fi++) {
        const auto& vinds = app.mesh->faces.row(fi);
        auto& ct = std::make_shared<TriangleConstraint>(vinds, app.mesh->verts, 0.9_r, 1.1_r, 100.0_r, cslist.size(), start_row);
        start_row += 2;
        cslist.emplace_back(ct);
    }
    
    // bending
    for (int vi = 0; vi < app.mesh->verts.rows(); vi++) {
        if (app.mesh->on_boundary(vi)) {} // boundary points, ignore
        else {
            const std::vector<int>& ring_idx = app.mesh->get_adjacency(vi);
            auto& ct = std::make_shared<BendingConstraint>(
                vi, ring_idx, app.mesh->verts, 0.01_r,
                cslist.size(), start_row);
            //for (auto& vi : ring_idx) std::cout << vi << " ";
            //std::cout << "\n";
            start_row += 1;
            cslist.emplace_back(ct);
        }
    }

    // pin
    std::unordered_set<int> pin_inds{
        //0, 1
        //1, 2
    };
    Matf_XX pin_v(pin_inds.size(), 3);
    int pin_idx = 0;
    for (int pi : pin_inds) {
        pin_v.row(pin_idx) = app.mesh->verts.row(pi);
        auto& ct = std::make_shared<PinConstraint>(
            pi, app.mesh->verts, 10000.0_r, cslist.size(), start_row);
        start_row += 1;
        cslist.emplace_back(ct);
        pin_idx++;
    }
    std::cout << pin_v << std::endl;

    app.solver->add_constraints(cslist);

    // nodal collision 
    Solver::NodalCollisionConstraintsList nodal_cslist;
    for (int vi = 0; vi < app.mesh->verts.rows(); vi++) {
        if (pin_inds.count(vi) == 0) {
            auto& ct = std::make_shared<NodalCollisionConstraint>(
                vi, app.mesh->verts, 100.0_r, cslist.size() + nodal_cslist.size(), start_row);
            nodal_cslist.emplace_back(ct);
            start_row += 1;
        }
    }
    app.solver->add_nodal_constraints(nodal_cslist);

    app.solver->m_nCDim = start_row;

    // obstacle
    app.solver->addObstacle(std::make_shared<SphereObstacle>(ADU::Vecf_3{ 0, -2.0_r, 0 }, 1.5_r));
    app.solver->addObstacle(std::make_shared<PlaneObstacle>(ADU::Vecf_3{ 0, -4.0_r, 0 }, 7.0_r, ADU::Vecf_3{ 0, 1, 0 }));

    // test D matrix
    const auto& constraints = app.solver->getConstraints();
    int m = constraints.size();
    Real ct_area = 0;
    for (int ci = 0; ci < m; ci++) {
        const auto& ct = constraints[ci];
        ct->test_D(app.mesh->verts);
    }
    

    app.solver->init(app.mesh->verts, app.mass, app.dt);

    polyscope::init();
    polyscope::registerSurfaceMesh("flat", app.mesh->verts, app.mesh->faces);
    //polyscope::registerCurveNetwork("boundary", app.positions0, boundary_faces);
    //polyscope::registerPointCloud("boundary pts", boundary_v);
    //polyscope::registerPointCloud("pin", pin_v); // ground
    //polyscope::registerPointCloud("points", std::vector<Eigen::Vector3f>{{0, 0, 0}}); // ground

    for (auto& o : app.solver->getObstacles()) {
        register_obstacle_mesh(o);
    }

    polyscope::view::resetCameraToHomeView();
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::state::lengthScale = 20;

    app.vis_meshP = polyscope::getSurfaceMesh("flat");
    
    polyscope::state::userCallback = main_loop;
    polyscope::options::groundPlaneEnabled = false;

    polyscope::show();

    std::cout << ADU::Timer::getLog();
}
