#pragma once

#include "mesh/mesh_container.h"
#include "mesh/mesh_phy_material.h"
#include "mediator/toml_to_config.h"

namespace ADU {

/// Load all meshes from APPConfig into MeshData. Unifies tri/tet/static logic.
struct MeshLoadResult {
    std::vector<size_t> mesh_id_list;
    std::vector<PhyxMaterial> material_list;
    int static_mesh_id_begin{};
    int static_vert_begin{};
    std::vector<std::pair<size_t, std::string>> mesh_animate_info;
};

inline MeshLoadResult load_meshes_from_config(
    MeshData& mesh_data,
    const APPConfig& config,
    const std::filesystem::path& resource_path)
{
    MeshLoadResult result;
    constexpr Real PI = 3.1415926_r;

    auto make_quat = [&PI](const APPConfig::Mesh::T& t) {
        Eigen::Quaternion<Real> q_x(Eigen::AngleAxis<Real>(PI * t.rotate.x() / 180.0_r, Vecf_3::UnitX()));
        Eigen::Quaternion<Real> q_y(Eigen::AngleAxis<Real>(PI * t.rotate.y() / 180.0_r, Vecf_3::UnitY()));
        Eigen::Quaternion<Real> q_z(Eigen::AngleAxis<Real>(PI * t.rotate.z() / 180.0_r, Vecf_3::UnitZ()));
        return q_z * q_y * q_x;
    };

    // First pass: tri and tet meshes
    for (const auto& cfg : config.meshes) {
        if (cfg.f.type != "tri" && cfg.f.type != "tet") continue;
        const std::string path = (resource_path / cfg.f.path).string();
        size_t mesh_id = 0;

        if (cfg.f.type == "tri") {
            mesh_id = mesh_data.add_mesh(path, MeshData::TRI, cfg.density, cfg.f.color_id);
            if (config.mesh_post.center_mesh_ids.count(static_cast<int>(mesh_id)))
                mesh_data.centralize(mesh_id);
            mesh_data.scale(cfg.t.scale, mesh_id);
            mesh_data.rotate(make_quat(cfg.t), mesh_id);
            mesh_data.translate(cfg.t.translate, mesh_id);
            result.mesh_id_list.push_back(mesh_id);
            result.material_list.push_back(
                PhyxMaterial(cfg.density, cfg.k.stretch, cfg.k.min_limit, cfg.k.max_limit, cfg.k.bending, cfg.k.use_limit));
        }
        else {
            mesh_id = mesh_data.add_mesh(path, MeshData::TET, cfg.density, cfg.f.color_id);
            if (config.mesh_post.center_mesh_ids.count(static_cast<int>(mesh_id)))
                mesh_data.centralize(mesh_id);
            mesh_data.rotate(make_quat(cfg.t), mesh_id);
            mesh_data.scale(cfg.t.scale, mesh_id);
            mesh_data.translate(cfg.t.translate, mesh_id);
            result.mesh_id_list.push_back(mesh_id);
            result.material_list.push_back(
                PhyxMaterial(cfg.density, cfg.k.stretch, cfg.k.min_limit, cfg.k.max_limit, cfg.k.use_limit));
        }
    }

    result.static_mesh_id_begin = static_cast<int>(result.mesh_id_list.size());
    result.static_vert_begin = static_cast<int>(mesh_data.verts.rows());

    // Second pass: static meshes
    for (const auto& cfg : config.meshes) {
        if (cfg.f.type != "static") continue;
        const std::string path = (resource_path / cfg.f.path).string();
        size_t mesh_id = mesh_data.add_mesh(path, MeshData::STATIC, 1e12_r, cfg.f.color_id);
        if (config.mesh_post.center_mesh_ids.count(static_cast<int>(mesh_id)))
            mesh_data.centralize(mesh_id);
        mesh_data.scale(cfg.t.scale, mesh_id);
        mesh_data.rotate(make_quat(cfg.t), mesh_id);
        mesh_data.translate(cfg.t.translate, mesh_id);
        result.mesh_id_list.push_back(mesh_id);
        result.material_list.push_back(
            PhyxMaterial(cfg.density, cfg.k.stretch, cfg.k.min_limit, cfg.k.max_limit, cfg.k.bending, cfg.k.use_limit));
        mesh_data.mesh_list[mesh_id]->is_static = true;
        if (!cfg.animate_path.empty())
            result.mesh_animate_info.push_back({ mesh_id, cfg.animate_path });
    }

    return result;
}

} // namespace ADU
