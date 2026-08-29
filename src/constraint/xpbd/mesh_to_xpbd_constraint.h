#pragma once

#include "constraint/xpbd/xpbd_constraint_factory.h"
#include "constraint/bending_constraint.h"
#include "mesh/mesh_container.h"
#include "mesh/mesh_phy_material.h"

namespace ADU {

/// Build the XPBD constraints of one mesh (all built-in kinds:
/// 1 tri stretch, 2 tet strain, 3 bending) into `out`.
inline void build_xpbd_constraints(MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
    std::vector<std::shared_ptr<XPBDConstraint>>& out) {
    for (int tid = 1; tid <= 3; tid++) {
        create_xpbd_constraints_for_type(tid, mesh_data, mesh_id, material, out);
    }
}

/// Built-in XPBD constraint creators (Macklin XPBD formulas):
/// 1 = FEMTriangleConstraint, 2 = XPBD_FEMTetConstraint, 3 = XPBDBendingConstraint.
/// Note: the bending creator derives rest-shape data from an ADU
/// BendingConstraint (mesh adjacency), exactly like the pre-archive code.
inline void ensure_builtin_xpbd_constraints_registered() {
    if (xpbd_detail::xpbd_builtin_registered()) return;
    xpbd_detail::xpbd_builtin_registered() = true;

    // type 1: FEM triangle stretch (cloth)
    register_xpbd_constraint_creator(1, [](MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
                                           std::vector<std::shared_ptr<XPBDConstraint>>& constraints) {
        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);
        if (mesh.type == MeshData::TRI) {
            const ADU::Real stiff = material.tri_stretch_k;
            for (int fi = mesh.start_faceIdx; fi < mesh.end_faceIdx; fi++) {
                const auto& vinds = mesh_data.faces.row(fi);
                auto xct = std::make_shared<FEMTriangleConstraint>();
                const bool res = xct->initConstraint(mesh_data.verts, vinds(0), vinds(1), vinds(2),
                    stiff, stiff, stiff, 0.3_r, 0.3_r);
                if (!res) continue;
                constraints.emplace_back(xct);
            }
        }
    });

    // type 2: FEM tet strain (soft body)
    register_xpbd_constraint_creator(2, [](MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
                                           std::vector<std::shared_ptr<XPBDConstraint>>& constraints) {
        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);
        if (mesh.type == MeshData::TET) {
            const ADU::Real stiff = material.tet_stretch_k;
            for (int ti = mesh.start_tetIdx; ti < mesh.end_tetIdx; ti++) {
                const auto& vinds = mesh_data.tets.row(ti);
                auto xct = std::make_shared<XPBD_FEMTetConstraint>();
                const bool res = xct->initConstraint(mesh_data.verts, vinds(0), vinds(1), vinds(2), vinds(3),
                    stiff, 0.4_r);
                if (!res) continue;
                constraints.emplace_back(xct);
            }
        }
    });

    // type 3: bending (quadratic mean curvature, derived from an ADU bending constraint)
    register_xpbd_constraint_creator(3, [](MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
                                           std::vector<std::shared_ptr<XPBDConstraint>>& constraints) {
        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);
        if (mesh.type == MeshData::TRI) {
            for (int vi = mesh.start_vertIdx; vi < mesh.end_vertIdx; vi++) {
                const std::vector<int>& ring_idx = mesh_data.get_adjacency(vi);
                if (ring_idx.size() < 3) continue;
                auto bc = std::make_shared<BendingConstraint>(
                    vi, ring_idx, mesh_data.verts, material.tri_bending_k, 0, 0);
                auto xct = std::make_shared<XPBDBendingConstraint>();
                if (!xct->initConstraint(bc)) continue;
                constraints.emplace_back(xct);
            }
        }
    });
}

} // namespace ADU
