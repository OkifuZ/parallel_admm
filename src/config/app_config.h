#pragma once

#include <toml++/toml.h>
#include <mutils/common_types.h>
#include "mutils/exception_handle.h"
#include <set>


template <typename T>
inline T _CTML(std::optional<T> x){
    if (x) return *x;
    else ADU::make_exception("CTML_ APPConfig toml file invalid");
}


struct APPConfig {
    std::string scene_name{};
    std::string out_file{};
    bool separate_out{ false };
    bool show_windows{ true };
    bool out_bin{ false };
    int end_frame{ 2000 };

    bool use_GPU{ false };



    struct Global {
        ADU::Real dt;
        std::vector<int> pin_ids;
        ADU::Real g;
        int sub_step{ 1 };
        //ADU::Real density;
    };

    Global global;

    // DEPRECATED: XPBD solver archived (archive/xpbd/). Fields kept only for
    // backward compatibility with old scene TOMLs; they are parsed but ignored.
    struct XPBD {
        bool use_XPBD{ false };
        int XPBD_iter{ 100 };
    };

    XPBD xpbd;

    struct Solver
    {
        struct ADMM {
            int admm_max_iter;
            int DCD_interval;
            int GS_max_iter;
            int Global_Jacobi_iter{20};
            bool warmstart_Uc;
            bool warmstart_Ue;
            bool dump_A{ false };
        };
        ADMM admm;

        struct Contact {
            bool enable;
            bool use_CCD;
            ADU::Real w_scale;
            ADU::Real kappa;
            ADU::Real beta;
            ADU::Real mu;
            bool unique;
            bool use_jacobi{false};

            bool coloring{ false };
            bool use_heu_wc{ false };
            ADU::Real wc_beta{ 25 };
            ADU::Real wc_sigma{ 0.001 };
        };
        Contact contact;

        struct Damp {
            ADU::Real kl;
            ADU::Real km;
        };
        Damp damp;
    };
    Solver solver;

    struct DCD {
        struct Narrow {
            int max_collision;
            ADU::Real thickness;
        };
        Narrow narrow;
        struct Broad {
            ADU::Real radius;
            std::string strategy;
        };
        Broad broad;
    };
    DCD dcd;

    struct Mesh {
        ADU::Real density{ 1 };
        struct F {
            std::string path;
            std::string type;
            int color_id{};
        };
        F f;
        struct T {
            ADU::Real scale{ 1.0 };
            ADU::Vecf_3 translate;
            ADU::Vecf_3 rotate;
        };
        T t;
        struct K {
            ADU::Real stretch;
            ADU::Real min_limit;
            ADU::Real max_limit;
            ADU::Real bending;
            bool use_limit{ false };
        };
        K k;

        std::string animate_path = "";

        //bool centeralize{ false };
    };
    std::vector<Mesh> meshes;

    struct Collider {
        std::string type;
        ADU::Vecf_3 center;
        ADU::Real halfside;
        ADU::Vecf_3 normal;
    };
    std::vector<Collider> colliders;

    struct Viewer {
        ADU::Real scale;
        ADU::Vecf_3 background_color;
        bool enable_ground{ false };
    };
    Viewer viewer;

    struct MeshPost {
        std::set<int> center_mesh_ids;
    };
    MeshPost mesh_post;

    


    void load_config(const std::string& toml_path) {
        using namespace toml;
        using namespace ADU;

        auto& app_config = *this;
        table config = parse_file(toml_path);
        app_config.scene_name = _CTML(config["scene_name"].value<std::string>());
        app_config.out_file = _CTML(config["out_file"].value<std::string>());
        app_config.separate_out = _CTML(config["separate_out"].value<bool>());
        app_config.show_windows = _CTML(config["show_windows"].value<bool>());
        auto out_bin_temp = config["out_bin"].value<bool>();
        if (out_bin_temp) app_config.out_bin = *out_bin_temp;

        auto end_frame_opt = config["end_frame"].value<int>();
        if (end_frame_opt && *end_frame_opt > 0) {
            end_frame = *end_frame_opt;
        }

        auto& global_config = config["global"];
        auto& xpbd_config = config["xpbd"];
        auto& solver_config = config["solver"];
        auto& DCD_config = config["DCD"];
        auto& mesh_config = config["mesh"];
        auto& mesh_post_config = config["mesh_post"];
        auto& collider_config = config["collider"];
        auto& viewer_config = config["viewer"];

        auto& useGPU = config["GPU"];
        if (useGPU) {
            app_config.use_GPU = *useGPU.value<bool>();
        }

        // global
        app_config.global.dt = _CTML(global_config["dt"].value<ADU::Real>());
        auto substep_tmp = global_config["sub_step"].value<int>();
        if (substep_tmp) {
            app_config.global.sub_step = *substep_tmp;
        }

        auto pin_arr_toml = *global_config["pin"].as_array();
        for (auto& pin : pin_arr_toml) {
            app_config.global.pin_ids.push_back(_CTML(pin.value<int>()));
        }
        auto x = global_config["g"];
        app_config.global.g = _CTML(global_config["g"].value<ADU::Real>());
        //app_config.global.density = _CTML(global_config["density"].value<ADU::Real>());

        if (xpbd_config) {
            app_config.xpbd.use_XPBD = _CTML(xpbd_config["enable"].value<bool>());
            app_config.xpbd.XPBD_iter = _CTML(xpbd_config["iter"].value<int>());
        }



        // solver
        app_config.solver.admm.admm_max_iter = _CTML(solver_config["admm_max_iter"].value<int>());
        app_config.solver.admm.DCD_interval = _CTML(solver_config["DCD_interval"].value<int>());
        app_config.solver.admm.GS_max_iter = _CTML(solver_config["GS_max_iter"].value<int>());
        auto global_iter = solver_config["Global_Jacobi_iter"];
        if (global_iter) {
            app_config.solver.admm.Global_Jacobi_iter = *global_iter.value<int>();
        }

        auto& solver_admm_config = solver_config["admm"];
        app_config.solver.admm.warmstart_Ue = _CTML(solver_admm_config["warmstart_Ue"].value<bool>());
        app_config.solver.admm.warmstart_Uc = _CTML(solver_admm_config["warmstart_Uc"].value<bool>());
        auto dump_A = solver_admm_config["dump_A"];
        if (dump_A) {
            app_config.solver.admm.dump_A = *dump_A.value<bool>();
        }

        auto& solver_contact_config = solver_config["contact"];
        app_config.solver.contact.enable = _CTML(solver_contact_config["enable"].value<bool>());
        app_config.solver.contact.use_CCD = _CTML(solver_contact_config["use_CCD"].value<bool>());
        auto use_jc = solver_contact_config["use_Jacobi"].value<bool>();
        if (use_jc) {
            app_config.solver.contact.use_jacobi = *use_jc;
        }
        app_config.solver.contact.w_scale = _CTML(solver_contact_config["w_scale"].value<ADU::Real>());
        app_config.solver.contact.mu = _CTML(solver_contact_config["mu"].value<ADU::Real>());
        app_config.solver.contact.kappa = _CTML(solver_contact_config["kappa"].value<ADU::Real>());
        app_config.solver.contact.beta = _CTML(solver_contact_config["beta"].value<ADU::Real>());
        app_config.solver.contact.unique = _CTML(solver_contact_config["unique"].value<bool>());

        auto coloring = solver_contact_config["coloring"].value<bool>();
        if (coloring) {
            app_config.solver.contact.coloring = *coloring;
        }

        auto use_wc = solver_contact_config["use_heu_wc"].value<bool>();
        if (use_wc) {
            app_config.solver.contact.use_heu_wc = *use_wc;
        }
        if (app_config.solver.contact.use_heu_wc) {
            auto heu_beta = solver_contact_config["wc_beta"].value<ADU::Real>();
            if (heu_beta) app_config.solver.contact.wc_beta = *heu_beta;
            auto heu_sigma = solver_contact_config["wc_sigma"].value<ADU::Real>();
            if (heu_sigma) app_config.solver.contact.wc_sigma = *heu_sigma;
        }


        auto& solver_damp_config = solver_config["damp"];
        app_config.solver.damp.kl = _CTML(solver_damp_config["kl"].value<ADU::Real>());
        app_config.solver.damp.km = _CTML(solver_damp_config["km"].value<ADU::Real>());

        // DCD
        auto& DCD_narrow_config = DCD_config["narrow"];
        auto& DCD_broad_config = DCD_config["broad"];
        app_config.dcd.narrow.max_collision = _CTML(DCD_narrow_config["max_collision"].value<int>());
        app_config.dcd.narrow.thickness = _CTML(DCD_narrow_config["thickness"].value<Real>());
        app_config.dcd.broad.radius = _CTML(DCD_broad_config["radius"].value<Real>());
        app_config.dcd.broad.strategy = _CTML(DCD_broad_config["strategy"].value<std::string>());

        // mesh
        auto& mesh_list_config = *mesh_config.as_array();
        for (int i = 0; i < mesh_list_config.size(); i++) {
            auto mesh = APPConfig::Mesh();
            auto& mesh_cfg = *mesh_list_config[i].as_table();


            mesh.f.path = _CTML(mesh_cfg["f"]["path"].value<std::string>());
            mesh.f.type = _CTML(mesh_cfg["f"]["type"].value<std::string>());
            mesh.f.color_id = _CTML(mesh_cfg["f"]["color"].value<int>());

            mesh.t.scale = _CTML(mesh_cfg["t"]["scale"].value<Real>());
            for (int i = 0; i < 3; i++) {
                mesh.t.translate(i) = _CTML(mesh_cfg["t"]["translate"][i].value<Real>());
            }
            for (int i = 0; i < 3; i++) {
                mesh.t.rotate(i) = _CTML(mesh_cfg["t"]["rotate"][i].value<Real>());
            }

            if (mesh.f.type == "tri") {
                mesh.density = _CTML(mesh_cfg["density"].value<ADU::Real>());
                mesh.k.stretch = _CTML(mesh_cfg["k"]["stretch"].value<Real>());
                mesh.k.min_limit = _CTML(mesh_cfg["k"]["min_limit"].value<Real>());
                mesh.k.max_limit = _CTML(mesh_cfg["k"]["max_limit"].value<Real>());
                mesh.k.bending = _CTML(mesh_cfg["k"]["bending"].value<Real>());
                mesh.k.use_limit = _CTML(mesh_cfg["k"]["use_limit"].value<bool>());
            }
            else if (mesh.f.type == "tet") {
                mesh.density = _CTML(mesh_cfg["density"].value<ADU::Real>());
                mesh.k.stretch = _CTML(mesh_cfg["k"]["stretch"].value<Real>());
                mesh.k.min_limit = _CTML(mesh_cfg["k"]["min_limit"].value<Real>());
                mesh.k.max_limit = _CTML(mesh_cfg["k"]["max_limit"].value<Real>());
                mesh.k.use_limit = _CTML(mesh_cfg["k"]["use_limit"].value<bool>());
            }
            else if (mesh.f.type == "static") {
                mesh.animate_path = _CTML(mesh_cfg["animate"].value<std::string>());
            }
            app_config.meshes.push_back(mesh);
        }

        // mesh post
        auto& center_ids_config = *mesh_post_config["center_ids"].as_array();
        for (int i = 0; i < center_ids_config.size(); i++) {
            app_config.mesh_post.center_mesh_ids.insert(_CTML(center_ids_config[i].value<int>()));
        }

        // collider
        auto& collider_list_config = *collider_config.as_array();
        for (int i = 0; i < collider_list_config.size(); i++) {
            auto collider = APPConfig::Collider();
            auto& collider_cfg = *collider_list_config[i].as_table();

            collider.type = _CTML(collider_cfg["type"].value<std::string>());
            if (collider.type == "plane") {
                for (int i = 0; i < 3; i++) {
                    collider.center(i) = _CTML(collider_cfg["center"][i].value<Real>());
                }
                collider.halfside = *collider_cfg["halfside"].value<Real>();
                for (int i = 0; i < 3; i++) {
                    collider.normal(i) = _CTML(collider_cfg["normal"][i].value<Real>());
                }
                app_config.colliders.push_back(collider);
            }
        }

        // viewer
        app_config.viewer.scale = _CTML(viewer_config["scale"].value<Real>());
        for (int i = 0; i < 3; i++) {
            app_config.viewer.background_color(i) = _CTML(viewer_config["color"]["background"][i].value<Real>());
        }
        app_config.viewer.enable_ground = _CTML(viewer_config["enable_ground"].value<bool>());
    }
};

