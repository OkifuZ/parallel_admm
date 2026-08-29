#pragma once

#include "mesh/mesh_container.h"
#include "mediator/toml_to_config.h"
#include "mutils/cformat.h"
#include "igl/writeOBJ.h"
#include "igl/writePLY.h"

#include <filesystem>
#include <string>

namespace ADU {

/// Export current frame to OBJ or PLY. Handles separate_out vs single-file modes.
inline void export_frame(
    const std::filesystem::path& out_path,
    const std::string& scene_name,
    size_t step_idx,
    const Matf_X3& all_verts,
    const Mati_X3& surface_tris,
    const MeshData* mesh_data,
    bool use_bin,
    bool separate_out)
{
    if (separate_out && mesh_data) {
        for (size_t mesh_id = 0; mesh_id < mesh_data->mesh_list.size(); mesh_id++) {
            std::string out_file_na = cformat("%s\\_%03zu\\_%05zu", scene_name.c_str(), mesh_id, step_idx);
            const auto& mesh_obj = mesh_data->mesh_list[mesh_id];
            const auto verts = all_verts.block(mesh_obj->start_vertIdx, 0,
                mesh_obj->end_vertIdx - mesh_obj->start_vertIdx, 3);
            const auto faces = mesh_data->getFaceInds(static_cast<int>(mesh_id));
            if (use_bin) {
                if (!igl::writePLY((out_path / out_file_na).string() + ".ply", verts, faces, igl::FileEncoding::Binary))
                    throw std::runtime_error("failed to write PLY file");
            }
            else {
                if (!igl::writeOBJ((out_path / out_file_na).string() + ".obj", verts, faces))
                    throw std::runtime_error("failed to write OBJ file");
            }
            std::cout << "saved to " << (out_path / out_file_na) << std::endl;
        }
    }
    else {
        std::string out_file_na = cformat("%s\\_%05zu", scene_name.c_str(), step_idx);
        if (use_bin) {
            if (!igl::writePLY((out_path / out_file_na).string() + ".ply", all_verts, surface_tris, igl::FileEncoding::Binary))
                throw std::runtime_error("failed to write PLY file");
        }
        else {
            if (!igl::writeOBJ((out_path / out_file_na).string() + ".obj", all_verts, surface_tris))
                throw std::runtime_error("failed to write OBJ file");
        }
        std::cout << "saved to " << (out_path / out_file_na) << std::endl;
    }
}

} // namespace ADU
