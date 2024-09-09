#include <igl/readOBJ.h>
#include <igl/readMSH.h>
#include <igl/boundary_facets.h>
#include <igl/adjacency_list.h>
#include <igl/unique.h>
#include <igl/volume.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "src_config.h"
#include "mesh/mesh_container.h"
#include "solver/admm_full_solver.h"
#include "constraint/constraint.h"
#include "constraint/tetrahedral_constraint.h"
#include "constraint/triangle_constraint.h"
#include "constraint/bending_constraint.h"
#include "constraint/pin_constraint.h"
#include "mutils/mesh_io.h"

#include "mutils/common_types.h"

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
    size_t mesh_id = app.mesh->add_mesh((resource_path / "tetmesh/octocat.msh").string(), MeshData::TET);

    app.mesh->scale(0.02_r, mesh_id);
    app.mesh->translate({ 0, 1.0_r, 0 }, mesh_id);

    app.mesh->precompute_geometry_info();

    app.mesh->get_mass(1.0_r, app.mass);

    app.solver = std::make_unique<ADMMSolverFull>();
    static_cast<ADMMSolverFull*>(app.solver.get())->warmstart_Ue = true;
    app.solver->parallel = true;

    Solver::ConstraintsList cslist;
    int start_row = 0;

    // tet strain
    for (int teti = 0; teti < app.mesh->tets.rows(); teti++) {
        const auto& tetinds = app.mesh->tets.row(teti);
        auto& ct = std::make_shared<TetrahedralConstraint>(tetinds, app.mesh->verts, 0.9_r, 1.1_r, 10000.0_r, cslist.size(), start_row);
        start_row += 3;
        cslist.emplace_back(ct);
    }

    // pin
    std::unordered_set<int> pin_inds{
        //79, 60, 20, 39
        //79, 40, 120, 159
         //7 ,0, 4, 3 
        //608, 2181
    };
    Matf_XX pin_v(pin_inds.size(), 3);
    int pin_idx = 0;
    for (int pi : pin_inds) {
        if (pi >= app.mesh->verts.rows()) continue;
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
    for (int vi = 0; vi < app.mesh->boundary_vinds_tris.rows(); vi++) {
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
    app.solver->addObstacle(std::make_shared<PlaneObstacle>(ADU::Vecf_3{ 0, -0.0_r, 0 }, 2.0_r, ADU::Vecf_3{ 0, 1, 0 }));

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
    polyscope::registerSurfaceMesh("flat", app.mesh->verts, app.mesh->boundary_tris);
    //polyscope::registerPointCloud("boundary pts", boundary_vx);
    //polyscope::registerCurveNetwork("boundary", app.positions0, boundary_faces);
    if (!pin_inds.empty()) {
        polyscope::registerPointCloud("pin", pin_v);
    }
    //polyscope::registerPointCloud("points", std::vector<Eigen::Vector3f>{{0, 0, 0}}); // ground

    for (auto& o : app.solver->getObstacles()) {
        register_obstacle_mesh(o);
    }

    polyscope::view::resetCameraToHomeView();
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::state::lengthScale = 5;

    app.vis_meshP = polyscope::getSurfaceMesh("flat");

    polyscope::state::userCallback = main_loop;
    polyscope::options::groundPlaneEnabled = false;

    polyscope::show();

    std::cout << "\n";
    std::cout << ADU::Timer::getLog() << std::endl;

}
