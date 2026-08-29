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
#include "constraint/tetrahedral_constraint.h"
#include "constraint/bending_constraint.h"
#include "constraint/pin_constraint.h"
#include "contact/broad_phase.h"

#include "constraint/mesh_to_constraint.h"
#include "contact/dcd_validation.h"

#include "mutils/exception_handle.h"
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
    std::shared_ptr<MeshData> mesh;
    Vecf_X mass;
    //Real dt = 0.0333_r / 3.0_r;
    Real dt = 0.0333_r;

    polyscope::SurfaceMesh* vis_meshP{};
    polyscope::PointCloud* vis_contactP{};

    unsigned int pause = 1;
    bool step = false;
    bool reset = false;
};

APP app;


void main_loop() {
    ImGui::CheckboxFlags("pause", (unsigned int*)&app.pause, ImGuiConfigFlags_NavEnableKeyboard); ImGui::SameLine();
    if (ImGui::Button("step")) app.step = true;

    if (ImGui::Button("reset")) app.reset = true;

    if (app.reset) {
        // reset solver only
        app.solver->reset(app.mesh->verts, false);
        app.mesh->clear_color();
        app.reset = false;
        app.pause = true;
    }

    if (!app.pause || app.step) {
        app.solver->step();
        app.step = false;
    }

    if (app.vis_meshP) {
        app.vis_meshP->updateVertexPositions(app.solver->getVertices());
        app.mesh->clear_color();
        auto contact_color = Vecf_3{ 0.890_r, 0.338_r, 0.11_r };
        for (const auto& ct : app.solver->prox_query->contact_info_list) {
            app.mesh->v_colors.row(ct.vinds[0]) = contact_color;
            app.mesh->v_colors.row(ct.vinds[1]) = contact_color;
            app.mesh->v_colors.row(ct.vinds[2]) = contact_color;
            app.mesh->v_colors.row(ct.vinds[3]) = contact_color;
        }
        app.vis_meshP->addVertexColorQuantity("contact", app.mesh->v_colors);

        //app.bp.updateBVH(app.solver->getVertices());
        //std::vector<std::vector<int>> candidates;
        //app.bp.query_point_triangle(app.solver->getVertices(), candidates);
        ///*auto& res = app.bp.query_point(Vecf_3{0, 1.0f, 0});*/
        //for (auto& i : candidates[134]) {
        //    app.mesh->f_colors.row(i) = Vecf_3{ 1.0_r, 0.0_r, 0.0_r };
        //}

        //app.vis_meshP->addFaceColorQuantity("bvh", app.mesh->f_colors);
    }
    if (app.vis_contactP) {
        app.vis_contactP->updatePointPositions(app.solver->prox_query->get_contact_points());
        app.vis_contactP->addVectorQuantity("normal", app.solver->prox_query->get_contact_normals());

    }
}


std::filesystem::path resource_path(PROJECT_RESOURCE_PATH);


int main() {

    // simulated mesh
    app.mesh = std::make_shared<MeshData>();

    //size_t mesh_id1 = app.mesh->add_mesh((resource_path / "frictionPD/ribbon.obj").string(), MeshData::TRI);
    //size_t mesh_id1 = app.mesh->add_mesh((resource_path / "trimesh/bowl.obj").string(), MeshData::TRI);
    size_t mesh_id1 = app.mesh->add_mesh((resource_path / "flat/flat_v134.obj").string(), MeshData::TRI);
    //size_t mesh_id1 = app.mesh->add_mesh((resource_path / "flat/flat.obj").string(), MeshData::TRI);
    //app.mesh->scale(2.0f, mesh_id1);
    //Eigen::Quaternion<ADU::Real> q(Eigen::AngleAxisd(3.1415926_r * 0.5_r, ADU::Vecf_3::UnitX()));
    //app.mesh->rotate(q, mesh_id1);
    //app.mesh->scale(0.6f, mesh_id1);
    app.mesh->scale(2.0f, mesh_id1);
    app.mesh->translate({ 0, 1.0_r, 0 }, mesh_id1);
    //app.mesh->translate({ 0, 4.0_r, 0 }, mesh_id1);
    auto material_id1 = PhyxMaterial(1.0_r, 1000.0_r, 0.95_r, 1.05_r, 1.0_r);

    //size_t mesh_id2 = app.mesh->add_mesh((resource_path / "tetmesh/octocat.msh").string(), MeshData::TET);
    //size_t mesh_id2 = app.mesh->add_mesh((resource_path / "tetmesh/cow_head.msh").string(), MeshData::TET);
    //size_t mesh_id2 = app.mesh->add_mesh((resource_path / "tetmesh/bar-186.msh").string(), MeshData::TET);
    //size_t mesh_id2 = app.mesh->add_mesh((resource_path / "tetmesh/middle_ball.msh").string(), MeshData::TET);
    //size_t mesh_id2 = app.mesh->add_mesh((resource_path / "tetmesh/tet-pyramid.msh").string(), MeshData::TET);
    //size_t mesh_id2 = app.mesh->add_mesh((resource_path / "tetmesh/tet.msh").string(), MeshData::TET);
    //app.mesh->centralize(mesh_id2);
    //app.mesh->scale(0.2_r, mesh_id2);
    //app.mesh->scale(8.0_r, mesh_id2);
    //app.mesh->scale(2.0_r, mesh_id2);
    //app.mesh->translate({ 0.0_r, 2.5_r, 0 }, mesh_id2);
    //app.mesh->translate({ 0, 2.5_r, 0 }, mesh_id2);
    //app.mesh->translate({ 0, 3.1_r, 0 }, mesh_id2);
    //auto material_id2 = PhyxMaterial(1.0_r, 100.0_r, 0.9_r, 1.1_r);

    size_t mesh_id3 = app.mesh->add_mesh((resource_path / "flat/flat_v134.obj").string(), MeshData::TRI);
    app.mesh->scale(1.8f, mesh_id3);
    app.mesh->translate({ 0, 2.5_r, 0 }, mesh_id3);
    auto material_id3 = PhyxMaterial(1.0_r, 100.0_r, 0.95_r, 1.05_r, 0.1_r);
    
    size_t mesh_id4 = app.mesh->add_mesh((resource_path / "flat/flat_v377.obj").string(), MeshData::TRI);
    app.mesh->scale(1.6f, mesh_id4);
    Eigen::Quaternion<ADU::Real> q4(Eigen::AngleAxisd(3.1415926_r * 0.25_r, ADU::Vecf_3::UnitY()));
    app.mesh->rotate(q4, mesh_id4);
    app.mesh->translate({ 0, 3.0_r, 0 }, mesh_id4);
    auto material_id4 = PhyxMaterial(1.0_r, 100.0_r, 0.95_r, 1.05_r, 0.1_r);

    size_t mesh_id5 = app.mesh->add_mesh((resource_path / "flat/flat_v377.obj").string(), MeshData::TRI);
    app.mesh->scale(1.4f, mesh_id5);
    Eigen::Quaternion<ADU::Real> q5(Eigen::AngleAxisd(3.1415926_r * 0.5_r, ADU::Vecf_3::UnitY()));
    app.mesh->rotate(q5, mesh_id5);
    app.mesh->translate({ 0, 3.5_r, 0 }, mesh_id5);
    auto material_id5 = PhyxMaterial(1.0_r, 100.0_r, 0.95_r, 1.05_r, 0.1_r);

    size_t mesh_id6 = app.mesh->add_mesh((resource_path / "flat/flat_v377.obj").string(), MeshData::TRI);
    app.mesh->scale(1.2f, mesh_id6);
    Eigen::Quaternion<ADU::Real> q6(Eigen::AngleAxisd(3.1415926_r * 0.75_r, ADU::Vecf_3::UnitY()));
    app.mesh->rotate(q6, mesh_id6);
    app.mesh->translate({ 0, 4.0_r, 0 }, mesh_id6);
    auto material_id6 = PhyxMaterial(1.0_r, 100.0_r, 0.95_r, 1.05_r, 0.1_r);

    app.mesh->precompute_geometry_info();

    
    

    /*auto& res = app.bp.query_point(app.mesh->verts.row(60));
    for (auto& i : res.prims) {
        std::cout << i << " ";
        app.mesh->f_colors.row(i) = Vecf_3{1.0_r, 0.0_r, 0.0_r};
    }*/


    if (!thickness_validation_check(*(app.mesh))) {
        ADU::make_exception("thickness_validation_check failed");
    }

    app.mesh->get_mass(0.1_r, app.mass);

    app.solver = std::make_unique<ADMMImplCPU>();
    ADMMImplCPU* solver = static_cast<ADMMImplCPU*>(app.solver.get());
    solver->enable_frictional_contact = true;
    solver->warmstart_Ue = true;
    solver->warmstart_Uc = true;

    // constraints
    Solver::ConstraintsList cslist;
    Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id1, material_id1, cslist);
    //Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id2, material_id2, cslist);
    Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id3, material_id3, cslist);
    Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id4, material_id4, cslist);
    Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id5, material_id5, cslist);
    Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id6, material_id6, cslist);

    // pin
    std::unordered_set<size_t> pin_inds{
        0, 1, 2, 3
    };
    Matf_XX pin_v(pin_inds.size(), 3);
    int pin_idx = 0;
    for (int pi : pin_inds) {
        pin_v.row(pin_idx) = app.mesh->verts.row(pi);
        auto& ct = std::make_shared<PinConstraint>(
            pi, app.mesh->verts, 10000.0_r, cslist.size(), Mesh2Constraint::curr_start_row);
        Mesh2Constraint::curr_start_row += 1;
        cslist.emplace_back(ct);
        pin_idx++;
    }

    app.solver->add_constraints(cslist);

    // nodal collision 
    Solver::NodalCollisionConstraintsList nodal_cslist;
    for (int vi = 0; vi < app.mesh->verts.rows(); vi++) {
        if (pin_inds.count(vi) == 0) {
            auto& ct = std::make_shared<NodalCollisionConstraint>(
                vi, app.mesh->verts, 100.0_r, cslist.size() + nodal_cslist.size(), Mesh2Constraint::curr_start_row);
            nodal_cslist.emplace_back(ct);
            Mesh2Constraint::curr_start_row += 1;
        }
    }

    app.solver->add_nodal_constraints(nodal_cslist);

    app.solver->m_nCDim = Mesh2Constraint::curr_start_row;

    
    // obstacle
    //app.solver->addObstacle(std::make_shared<SphereObstacle>(ADU::Vecf_3{ 0, -2.0_r, 0 }, 1.5_r));
    app.solver->addObstacle(std::make_shared<PlaneObstacle>(ADU::Vecf_3{ 0, -2.0_r, 0 }, 5.0_r, ADU::Vecf_3{ 0, 1, 0 }));

    // test D matrix
    const auto& constraints = app.solver->getConstraints();
    int m = constraints.size();
    Real ct_area = 0;
    for (int ci = 0; ci < m; ci++) {
        const auto& ct = constraints[ci];
        ct->test_D(app.mesh->verts);
    }

    // proximal query 
    app.solver->prox_query = std::make_unique<ProximalQuery>(app.mesh);

    // TODO: make this to init
    app.solver->prox_query->broad_phase = std::make_shared<BroadPhase>(BroadPhase::REBUILD);
    app.solver->prox_query->broad_phase->mesh = app.mesh;
    app.solver->prox_query->broad_phase->buildBVH();




    app.solver->init(app.mesh->verts, app.mass, app.dt);

    polyscope::init();
    /*ADU::Mati_X3 tris(app.mesh->faces.rows() + app.mesh->boundary_tris.rows(), 3);
    tris << app.mesh->faces, app.mesh->boundary_tris;
    polyscope::registerSurfaceMesh("tri", app.mesh->verts, tris);*/
    polyscope::registerSurfaceMesh("surface", app.mesh->verts, app.mesh->surface_tris);
    polyscope::registerPointCloud("contact", app.solver->prox_query->get_contact_points());

    if (!pin_inds.empty()) polyscope::registerPointCloud("pin", pin_v); // ground

    polyscope::view::resetCameraToHomeView();
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::state::lengthScale = 10;

    app.vis_meshP = polyscope::getSurfaceMesh("surface");
    app.vis_contactP = polyscope::getPointCloud("contact");
    
    polyscope::state::userCallback = main_loop;
    polyscope::options::groundPlaneEnabled = false;
    polyscope::view::bgColor = { 0.1f, 0.1f, 0.1f, 0.0f };
    //polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    polyscope::show();

    std::cout << ADU::Timer::getLog();
}

