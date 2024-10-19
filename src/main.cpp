#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
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
#include "solver/admm_full_solver.h"
#include "solver/PBD_solver.h"
#include "solver/XPBD_solver.h"

#include "constraint/constraint.h"
#include "constraint/triangle_constraint.h"
#include "constraint/tetrahedral_constraint.h"
#include "constraint/bending_constraint.h"
#include "constraint/pin_constraint.h"
#include "contact/broad_phase.h"

#include "solver/animator.h"

#include "mediator/mesh_to_constraint.h"
#include "mediator/DCD_validation_check.h"
#include "mediator/toml_to_config.h"

#include "mutils/exception_handle.h"
#include "mutils/cformat.h"
#include "mutils/common_types.h"
#include "mutils/timer.h"
#include "mutils/aux_color.h"

#include "Eigen/Core"

#include "Eigen/Core"

#include <toml++/toml.h>
#include <argparse/argparse.hpp>

#include <string>
#include <filesystem>
#include <memory>
#include <unordered_set>

#include <fstream>
#include <functional>
#include"mutils/cnpy.h"

#include "mutils/cuda_tools.cuh"
#include "device_launch_parameters.h"
#include "contact/lbvh/lbvh.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "mutils/dist_g.cuh"

#include "contact/collision_detection_cuda.h"

bool enable_large_scale = false;

using namespace ADU;


class APP {
public:
    std::string name{};
    std::unique_ptr<Solver> solver;
    std::unique_ptr<XPBDSolver> XPBD_solver;

    int end_frame = 2000;


    std::shared_ptr<MeshData> mesh;
    Vecf_X mass;
    Real dt = 0.0333_r;
    int sub_step = 1;

    bool enable_XPBD{ false };
    int XPBD_iter{ 100 };

    int pin_start{};
    int pin_end{};
    Matf_XX pin_v; // for visualize

    polyscope::SurfaceMesh* vis_meshP{};
    polyscope::PointCloud* vis_contactP{};
    polyscope::PointCloud* pin_pts{};
    polyscope::CurveNetwork* vis_bvh{};
    polyscope::CurveNetwork* vis_eect{};


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


    BVH_GPU* bvh;

};

APP app;
APPConfig app_config;

std::filesystem::path out_path;

size_t step_idx = 0;


void update_bvh_draw() {
    if (!app.show_windows) return;
    int num = (app.bvh->bvs.size()) / 2;
    app.bvh_nodes.clear();
    app.bvh_edges.clear();

    for (int j = 0; j < app.bvh->bvs.size(); j++) {
        int i = j;
        float ox, oy, oz, bwidth, bheight, blength;
        ox = (app.bvh->bvs[i].lower.x);
        oy = (app.bvh->bvs[i].lower.y);
        oz = (app.bvh->bvs[i].lower.z);
        bwidth = (app.bvh->bvs[i].upper.x - app.bvh->bvs[i].lower.x);
        bheight = (app.bvh->bvs[i].upper.y - app.bvh->bvs[i].lower.y);
        blength = (app.bvh->bvs[i].upper.z - app.bvh->bvs[i].lower.z);

        app.bvh_nodes.push_back(glm::vec3{ ox,          oy,            oz });
        app.bvh_nodes.push_back(glm::vec3{ ox,          oy,            oz + blength });
        app.bvh_nodes.push_back(glm::vec3{ ox,          oy + bheight , oz });
        app.bvh_nodes.push_back(glm::vec3{ ox + bwidth, oy,            oz });
        app.bvh_nodes.push_back(glm::vec3{ ox + bwidth, oy + bheight , oz });
        app.bvh_nodes.push_back(glm::vec3{ ox + bwidth, oy ,           oz + blength });
        app.bvh_nodes.push_back(glm::vec3{ ox ,         oy + bheight , oz + blength });
        app.bvh_nodes.push_back(glm::vec3{ ox + bwidth, oy + bheight , oz + blength });

        size_t start = i * 8;
        app.bvh_edges.push_back({ start + 0, start + 1 });
        app.bvh_edges.push_back({ start + 0, start + 2 });
        app.bvh_edges.push_back({ start + 0, start + 3 });
        app.bvh_edges.push_back({ start + 2, start + 4 });
        app.bvh_edges.push_back({ start + 2, start + 6 });
        app.bvh_edges.push_back({ start + 3, start + 4 });
        app.bvh_edges.push_back({ start + 3, start + 5 });
        app.bvh_edges.push_back({ start + 4, start + 7 });
        app.bvh_edges.push_back({ start + 5, start + 1 });
        app.bvh_edges.push_back({ start + 5, start + 7 });
        app.bvh_edges.push_back({ start + 6, start + 1 });
        app.bvh_edges.push_back({ start + 6, start + 7 });
    }
}


void main_loop() {
    auto solver = (ADMMSolverFull_RL_damping*)(app.solver.get());
    
    ImGui::CheckboxFlags("pause", (unsigned int*)&app.pause, ImGuiConfigFlags_NavEnableKeyboard); ImGui::SameLine();
    if (ImGui::Button("step")) app.step = true;
    if (ImGui::Button("reset")) app.reset = true;
    ImGui::CheckboxFlags("export obj", (unsigned int*)&app.export_obj, ImGuiConfigFlags_NavEnableKeyboard); ImGui::SameLine();
    

    if (app.reset) {
        // reset solver only
        app.solver->reset(app.mesh->verts, false);
        app.mesh->clear_color();
        app.reset = false;
        app.pause = true;
        step_idx = 0;
    }

    if (!app.pause || app.step) {
        printf("step frame %d\n", step_idx);
        
        app.step = false;

        if (app.enable_XPBD && app.XPBD_solver) {
            app.XPBD_solver->step(); // substep inside
        }
        else {
            for (int i = 0; i < app.sub_step; i++ ) app.solver->step();
        }

        auto solver_ptr = static_cast<ADMMSolverFull_RL_damping*>(app.solver.get());

        //app.bvh->update();
        update_bvh_draw();

        if (app.export_obj && app.save_res) {
            if (app_config.seperate_out) {
                for (int mesh_id = 0; mesh_id < app.mesh->mesh_list.size(); mesh_id++) {
                    std::string out_file_na = cformat("%s\_%03d\_%05d", app_config.scene_name.c_str(), mesh_id, step_idx);
                    const auto& mesh_obj = app.mesh->mesh_list[mesh_id];
                    const auto& verts = app.solver->getVertices().block(mesh_obj->start_vertIdx, 0, mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3);
                    const auto& faces = app.mesh->getFaceInds(mesh_id);
                    if (app.use_bin) {
                        if (!igl::writePLY((out_path / out_file_na).string() + ".ply", verts, faces, igl::FileEncoding::Binary) ) {
                            throw std::runtime_error("failed to write PLY file");
                        }
                    }
                    else {
                        if (!igl::writeOBJ((out_path / out_file_na).string() + ".obj", verts, faces)) {
                            throw std::runtime_error("failed to write OBJ file");
                        }
                    }
                    std::cout << "saved to " << out_path / out_file_na << std::endl;
                }
            }
            else {
                std::string out_file_na = cformat("%s\_%05d", app_config.scene_name.c_str(), step_idx);
                if (app.use_bin) {
                    if (!igl::writePLY((out_path / out_file_na).string() + ".ply", app.solver->getVertices(), app.mesh->surface_tris, igl::FileEncoding::Binary)) {
                        throw std::runtime_error("failed to write PLY file");
                    }
                }
                else {
                    if (!igl::writeOBJ((out_path / out_file_na).string() + ".obj", app.solver->getVertices(), app.mesh->surface_tris)) {
                        throw std::runtime_error("failed to write OBJ file");
                    }
                }
                std::cout << "saved to " << out_path / out_file_na << std::endl;
            }
        }

        if (app.show_windows && app.vis_meshP) {
            app.vis_meshP->addVertexVectorQuantity("velocity", app.solver->getVelocities());
        }

        if (app.show_windows && app.end_of_step_callback) {
            app.end_of_step_callback();
        }

        step_idx += 1;
    }

    if (app.show_windows && app.vis_meshP) {
        app.vis_meshP->updateVertexPositions(app.solver->getVertices());
        app.mesh->clear_color();
    }
    if (app.show_windows && app.vis_contactP) {
        auto solver_ptr = static_cast<ADMMSolverFull_RL_damping*>(app.solver.get());
        auto& s_vec = solver_ptr->prox_query->get_SaSb_mean();
        app.vis_contactP->updatePointPositions(app.solver->prox_query->get_contact_points());
        app.vis_contactP->addVectorQuantity("normal", app.solver->prox_query->get_contact_normals());
        app.vis_contactP->addVectorQuantity("S", s_vec);

        /*if (!app.pause) {
            std::cout << step_idx << "\n";

            for (auto& ct: app.solver->prox_query->contact_info_list ) {
                std::cout << ct.vinds[0] << "-" << ct.vinds[1] << "; " << ct.vinds[2] << "-" << ct.vinds[3] << "\n" ;
            }
        }*/


    }
    if (app.show_windows ) {
        app.vis_bvh->updateNodePositions(app.bvh_nodes);
    }


    if (step_idx > app.end_frame) {
        app.pause = true;
        polyscope::unshow();
    }
}


int main(int argc, const char* argv[]) {

#ifdef ADMM_DEBUG
    std::filesystem::path resource_path(PROJECT_RESOURCE_PATH);
    //std::string config_path = (resource_path / "scene/test_XPBD_ADMM.toml").string();
#else
    argparse::ArgumentParser program("test");
    program.add_argument("config_file_path").help("path of scene config file");
    program.add_argument("resource_file_path").help("path of dir of resource");
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string config_path = program.get<std::string>("config_file_path");
    std::string resource_path_str = program.get<std::string>("resource_file_path");
    std::filesystem::path resource_path(resource_path_str);
#endif


    Eigen::setNbThreads(12);
    int n = Eigen::nbThreads();
    Eigen::initParallel();

    app_config.load_config(config_path);

    app.name = app_config.scene_name;

    out_path = std::filesystem::path(app_config.out_file);
    if (!std::filesystem::exists(out_path)) {
        try {
            std::filesystem::create_directories(out_path);
            app.save_res = true;
        }
        catch (const std::exception& e) {
            app.save_res = false;
        }
    }
    else {
        app.save_res = true;
    }

    app.use_bin = app_config.out_bin;
    app.show_windows = app_config.show_windows;
    app.end_frame = app_config.end_frame;
    app.sub_step = app_config.global.sub_step;
    app.dt = app_config.global.dt / app.sub_step;

    // simulated mesh
    app.mesh = std::make_shared<MeshData>();

    app.enable_XPBD = app_config.xpbd.use_XPBD;
    app.XPBD_iter = app_config.xpbd.XPBD_iter;

    std::vector<size_t> mesh_id_list;
    std::vector<PhyxMaterial> material_list;
    int static_mesh_id_begin;
    int static_vert_begin;
    printf("loading mesh...\n");
    for (int i = 0; i < app_config.meshes.size(); i++) {
        auto& mesh = app_config.meshes[i];
        if (mesh.f.type == "tri") {
            size_t mesh_id = app.mesh->add_mesh((resource_path / mesh.f.path).string(), MeshData::TRI, mesh.density, mesh.f.color_id);
            if (app_config.mesh_post.center_mesh_ids.count(mesh_id)) app.mesh->centerlize(mesh_id);
            app.mesh->scale(mesh.t.scale, mesh_id);
            Eigen::Quaternion<ADU::Real> q_x(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.x() / 180.0_r, ADU::Vecf_3::UnitX()));
            Eigen::Quaternion<ADU::Real> q_y(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.y() / 180.0_r, ADU::Vecf_3::UnitY()));
            Eigen::Quaternion<ADU::Real> q_z(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.z() / 180.0_r, ADU::Vecf_3::UnitZ()));
            app.mesh->roatate(q_z * q_y * q_x, mesh_id);
            app.mesh->translate(mesh.t.translate, mesh_id);
            auto material = PhyxMaterial(mesh.density, mesh.k.stretch, mesh.k.min_limit, mesh.k.max_limit, mesh.k.bending, mesh.k.use_limit);
            mesh_id_list.push_back(mesh_id);
            material_list.push_back(material);
        }
        else if (mesh.f.type == "tet") {
            size_t mesh_id = app.mesh->add_mesh((resource_path / mesh.f.path).string(), MeshData::TET, mesh.density, mesh.f.color_id);
            if (app_config.mesh_post.center_mesh_ids.count(mesh_id)) app.mesh->centerlize(mesh_id);
            Eigen::Quaternion<ADU::Real> q_x(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.x() / 180.0_r, ADU::Vecf_3::UnitX()));
            Eigen::Quaternion<ADU::Real> q_y(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.y() / 180.0_r, ADU::Vecf_3::UnitY()));
            Eigen::Quaternion<ADU::Real> q_z(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.z() / 180.0_r, ADU::Vecf_3::UnitZ()));
            app.mesh->roatate(q_z * q_y * q_x, mesh_id);
            app.mesh->scale(mesh.t.scale, mesh_id);
            app.mesh->translate(mesh.t.translate, mesh_id);
            auto material = PhyxMaterial(mesh.density, mesh.k.stretch, mesh.k.min_limit, mesh.k.max_limit, mesh.k.use_limit);
            mesh_id_list.push_back(mesh_id);
            material_list.push_back(material);
        }
    }
    printf("done loading mesh\n");

    static_mesh_id_begin = mesh_id_list.size();
    static_vert_begin = app.mesh->verts.rows();
    std::vector<std::pair<size_t, std::string>> mesh_animate_info;
    for (int i = 0; i < app_config.meshes.size(); i++) {
        auto& mesh = app_config.meshes[i];
        if (mesh.f.type == "static") {
            size_t mesh_id = app.mesh->add_mesh((resource_path / mesh.f.path).string(), MeshData::STATIC, 1e12_r, mesh.f.color_id);
            if (app_config.mesh_post.center_mesh_ids.count(mesh_id)) app.mesh->centerlize(mesh_id);
            app.mesh->scale(mesh.t.scale, mesh_id);
            Eigen::Quaternion<ADU::Real> q_x(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.x() / 180.0_r, ADU::Vecf_3::UnitX()));
            Eigen::Quaternion<ADU::Real> q_y(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.y() / 180.0_r, ADU::Vecf_3::UnitY()));
            Eigen::Quaternion<ADU::Real> q_z(Eigen::AngleAxis<ADU::Real>(3.1415926_r * mesh.t.rotate.z() / 180.0_r, ADU::Vecf_3::UnitZ()));
            app.mesh->roatate(q_z * q_y * q_x, mesh_id);
            app.mesh->translate(mesh.t.translate, mesh_id);
            auto material = PhyxMaterial(mesh.density, mesh.k.stretch, mesh.k.min_limit, mesh.k.max_limit, mesh.k.bending, mesh.k.use_limit);
            mesh_id_list.push_back(mesh_id);
            material_list.push_back(material);
            app.mesh->mesh_list[mesh_id]->is_static = true;
            if (mesh.animate_path != "") mesh_animate_info.push_back({ mesh_id, mesh.animate_path });
        }
    }

#ifdef PRINT_AABB
    std::cout << "mesh list size: " << app.mesh->mesh_list.size() << "\n";
    app.mesh->compuite_AABB();
    app.mesh->print_AABB();
#endif // PRINT_AABB

    app.mesh->precompute_geometry_info();
    app.mesh->static_mesh_id_begin = static_mesh_id_begin;
    app.mesh->static_vert_begin = static_vert_begin;

    std::unordered_set<int> surf_vinds_set;
    for (int i = 0; i < app.mesh->surface_vinds.rows(); i++) {
        surf_vinds_set.insert(app.mesh->surface_vinds(i));
    }

    ADU::Veci_X is_static(app.mesh->verts.rows());
    for (auto& m : app.mesh->mesh_list) {
        for (int i = m->start_vertIdx; i < m->end_vertIdx; i++) {
            is_static(i) = int(m->is_static);
        }
    }
    app.mesh->is_static = is_static;

    int mi = 0;
    for (auto& m : app.mesh->mesh_list) {
        printf("mesh %d: %d-%d\n", mi++, m->start_vertIdx, m->end_vertIdx - 1);
        auto ccc = CoolColor::color(m->color_id);
        printf("(%f, %f, %f)\n", ccc.x(), ccc.y(), ccc.z());
    }

    printf("tet number: %d\n", app.mesh->tets.rows());

    for (int i = 0; i < app.mesh->boundary_vinds_edges.rows(); i++) {
        if (i % 2 == 0) std::cout << app.mesh->boundary_vinds_edges.row(i) << ", ";
    }
    std::cout << "\n";

    app.mesh->get_mass(app.mass);

    // app.solver = std::make_unique<ADMMSolverFull_RL_damping>();
    // ADMMSolverFull_RL_damping* solver = static_cast<ADMMSolverFull_RL_damping*>(app.solver.get());
    app.solver = std::make_unique<ADMMParallelSolver>();
    ADMMParallelSolver* solver = static_cast<ADMMParallelSolver*>(app.solver.get());

    solver->static_mesh_id_begin = static_mesh_id_begin;
    solver->static_vert_begin = static_vert_begin;

    solver->vinds_surf_set = surf_vinds_set;
    solver->enable_frictional_contact = app_config.solver.contact.enable;
    solver->warmstart_Ue = app_config.solver.admm.warmstart_Ue;
    solver->warmstart_Uc = app_config.solver.admm.warmstart_Uc;
    solver->admm_max_iter = app_config.solver.admm.admm_max_iter;
    solver->collision_detection_interval = app_config.solver.admm.DCD_interval;
    solver->gs_max_iter = app_config.solver.admm.GS_max_iter;
    //solver->contact_w = app_config.solver.contact.contact_w;
    ADU::Real w_scale = app_config.solver.contact.w_scale;
    solver->kappa = app_config.solver.contact.kappa;
    solver->beta = app_config.solver.contact.beta;
    solver->mu = app_config.solver.contact.mu;
    solver->g = app_config.global.g;

    solver->use_CCD = app_config.solver.contact.use_CCD;

    solver->use_jacobi = app_config.solver.contact.use_jacobi;

    solver->coloring_parallel_contact = false;

    solver->damp_k_L = app_config.solver.damp.kl;
    solver->damp_k_M = app_config.solver.damp.km;

    solver->m_vStatic = is_static;

    solver->use_unique_contact = app_config.solver.contact.unique;

    // W_c
    solver->contact_w_list.resize(app.mesh->verts.rows());
    solver->contact_w_inv_list.resize(app.mesh->verts.rows());
    solver->contact_w_list.setConstant(1e12_r);
    solver->contact_w_inv_list.setZero();
    if (app_config.solver.contact.use_heu_wc) {
        solver->use_heuristic_W_c = true;
        solver->heu_sigma = app_config.solver.contact.wc_sigma;
        solver->heu_beta = app_config.solver.contact.wc_beta;
    }
    else {
        for (int vi = 0; vi < app.mesh->static_vert_begin; vi++) {
            solver->contact_w_list(vi) = app.mass(vi) / (app.dt * app.dt) * w_scale;
            solver->contact_w_inv_list(vi) = 1.0_r / solver->contact_w_list(vi);
        }
    }

    // constraints
    Solver::ConstraintsList cslist;
    Solver::XPBDConstraintList XPBD_cslist;
    for (int i = 0; i < mesh_id_list.size(); i++) {
        Mesh2Constraint::geometry_to_constraints(*app.mesh, mesh_id_list[i], material_list[i], cslist, XPBD_cslist);
    }

    // sum to nVerts
    // pin 
    if (app_config.scene_name == "rod_twist") {
        std::vector<int> pin_inds;
        for (int i = 0; i < 4; i++) {
            for (auto& pi : app_config.global.pin_ids) {
                pin_inds.push_back(app.mesh->get_mesh(i).start_vertIdx + pi);
            }
        }
        app_config.global.pin_ids = pin_inds;
    }
    app.solver->addPins(app_config.global.pin_ids);

    app.pin_v.resize(app_config.global.pin_ids.size(), 3); // for visualize
    Mesh2Constraint::pin_to_constraints(*app.mesh, app_config.global.pin_ids, cslist, app.pin_v, is_static, app.pin_start, app.pin_end);

    app.solver->add_constraints(cslist);
    app.solver->add_XPBDConstraints(XPBD_cslist);

    app.solver->m_nCDim = Mesh2Constraint::curr_start_row;

    // proximal query 
    ContactParameter::set_broadphase_radius(app_config.dcd.broad.radius);
    ContactParameter::set_thickness(app_config.dcd.narrow.thickness);

    app.solver->prox_query = std::make_unique<ProximalQuery>(app.mesh, app_config.dcd.narrow.max_collision);
    BroadPhase_embree::BVH_TYPE bvh_type;

    if (app_config.dcd.broad.strategy == "rebuild") bvh_type = BroadPhase_embree::BVH_TYPE::REBUILD;
    if (app_config.dcd.broad.strategy == "refit") bvh_type = BroadPhase_embree::BVH_TYPE::REFIT;
    else bvh_type = BroadPhase_embree::BVH_TYPE::REBUILD;

    app.solver->prox_query->broad_phase = std::make_shared<BroadPhase_embree>(bvh_type);
    app.solver->prox_query->broad_phase->mesh = app.mesh;
    app.solver->prox_query->broad_phase->build();

    /*app.solver->prox_query->broad_phase = std::make_shared<BroadPhase_simpleBVH>();
    app.solver->prox_query->broad_phase->mesh = app.mesh;
    app.solver->prox_query->broad_phase->build();*/

    if (!thickness_validation_check(*(app.mesh))) {
        ADU::make_exception("thickness_validation_check failed", false);
    }

    // obstacle
    for (int i = 0; i < app_config.colliders.size(); i++) {
        app.solver->addObstacle(
            std::make_shared<PlaneObstacle>(
                app_config.colliders[i].center,
                app_config.colliders[i].halfside,
                app_config.colliders[i].normal));
    }

    app.solver->animator = std::make_unique<ScriptAnimator>();
    app.solver->animator->mesh = app.mesh;
    for (const auto& p : mesh_animate_info) {
        app.solver->animator->attach_animate2mesh(p.first, (resource_path / p.second).string());
    }

    app.solver->init(app.mesh->verts, app.mass, app.dt);

    if (app.enable_XPBD) {
        printf("enable XPBD, initializing...\n");
        app.XPBD_solver = std::make_unique<XPBDSolver>();
        printf("xpbd iter: %d\n", app.XPBD_iter);
        app.XPBD_solver->init(app.solver.get(), app.XPBD_iter, app.sub_step);
        app.XPBD_solver->contact_stiffness = app_config.solver.contact.w_scale;
    }



    BVH_GPU bvh;
    app.bvh = &bvh;
    bvh.solver = app.solver.get();
    app.solver->bvh = &bvh;

    bvh.mesh = app.mesh.get();
    /*bvh.init(); 
    bvh.construct(true);*/

    app.bvh->init();
    app.bvh->construct(true);
    update_bvh_draw();
    

    polyscope::init();

    if (app.show_windows) {
        // viewer
        polyscope::registerSurfaceMesh("surface", app.mesh->verts, app.mesh->surface_tris);
        polyscope::registerPointCloud("contact", app.solver->prox_query->get_contact_points());

        polyscope::registerCurveNetwork("bvh", app.bvh_nodes, app.bvh_edges);
        app.vis_bvh = polyscope::getCurveNetwork("bvh");

       // polyscope::registerCurveNetwork("ee ct", app.mesh->verts, app.ee_edges);
       // app.vis_eect = polyscope::getCurveNetwork("ee ct");

        if (!app_config.global.pin_ids.empty()) {
            polyscope::registerPointCloud("pin", app.pin_v); // ground
            app.pin_pts = polyscope::getPointCloud("pin");
        }

        polyscope::view::resetCameraToHomeView();
        polyscope::options::automaticallyComputeSceneExtents = false;
        polyscope::state::lengthScale = app_config.viewer.scale;

        app.vis_meshP = polyscope::getSurfaceMesh("surface");
        app.vis_contactP = polyscope::getPointCloud("contact");

        app.vis_meshP->addFaceColorQuantity("mesh color", app.mesh->f_colors);

        polyscope::options::groundPlaneEnabled = app_config.viewer.enable_ground;
        polyscope::view::bgColor = { 
            (float)app_config.viewer.background_color.x(), 
            (float)app_config.viewer.background_color.x(), 
            (float)app_config.viewer.background_color.x(), 
            (float)0.0_r};
    }
    polyscope::state::userCallback = main_loop;
    polyscope::show();
   
    std::cout << "max collision num: " << solver->prox_query->hist_max_collision_num << "\n";
    std::cout << "logger:\n";
    std::cout << ADU::Timer::getLog();
    

}
