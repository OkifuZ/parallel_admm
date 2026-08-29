#include "constraint/constraint.h"
#include "constraint/bending_constraint.h"
#include "constraint/triangle_constraint.h"
#include "constraint/tetrahedral_constraint.h"
#include "constraint/pin_constraint.h"
#include "constraint/nodal_collision_constraint.h"
#include "constraint/constraint_factory.h"

#include "mesh/mesh_container.h"
#include "mesh/mesh_phy_material.h"

#include "mutils/common_types.h"

#include <vector>

//#define DISABLE_BENDING
//#define DISABLE_STRETCHING

class Mesh2Constraint {
public:
    inline static size_t curr_start_row = 0;

    std::vector<int> invalid_inds;

    /// Build the constraints of one mesh for one type id through the
    /// constraint-factory registry (see constraint/constraint_factory.h).
    /// Unknown type ids are ignored (returns without error), so solvers can
    /// request a fixed set of constraint kinds regardless of the mesh type.
    static void geometry_to_constraints(
        MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material, 
        std::vector<std::shared_ptr<Constraint>>& constraints, int typeID)
    {
        ADU::create_constraints_for_type(typeID, mesh_data, mesh_id, material, constraints);
    }

    static void pin_to_constraints(MeshData& mesh_data, const std::vector<int>& pin_inds, 
        std::vector<std::shared_ptr<Constraint>>& constraints, ADU::Matf_XX& pin_v, ADU::Veci_X& is_static, int& pin_start, int& pin_end) 
    {
        using namespace ADU;
        pin_v.resize(pin_inds.size(), 3);
        pin_start = constraints.size();
        int pin_idx = 0;
        for (int pi : pin_inds) {
            if (is_static(pi)) continue;
            pin_v.row(pin_idx) = mesh_data.verts.row(pi);
            auto& ct = std::make_shared<PinConstraint>(
                pi, mesh_data.verts, 100000.0_r, constraints.size(), Mesh2Constraint::curr_start_row);
            Mesh2Constraint::curr_start_row += 1;
            constraints.emplace_back(ct);
            pin_idx++;
        }
        pin_end = constraints.size();
    }

    static void nodal_to_constraints(MeshData& mesh_data, const std::vector<int>& pin_inds, 
        const std::vector<std::shared_ptr<Constraint>>& cslist, 
        std::vector<std::shared_ptr<NodalCollisionConstraint>>& nodal_cslist) 
    {
        using namespace ADU;
        std::unordered_set<int> pin_inds_set(pin_inds.begin(), pin_inds.end());
        //printf("%d\n", pin_inds.size());
        for (int vi = 0; vi < mesh_data.static_vert_begin; vi++) {
            if (pin_inds_set.count(vi) == 0) {
                auto& ct = std::make_shared<NodalCollisionConstraint>(
                    vi, mesh_data.verts, 100.0_r, cslist.size() + nodal_cslist.size(), Mesh2Constraint::curr_start_row);
                nodal_cslist.emplace_back(ct);
                Mesh2Constraint::curr_start_row += 1;
            }
        }
    }

    static void cluster_constraints(const std::vector<std::shared_ptr<Constraint>>& cslist, std::vector<std::shared_ptr<Constraint>>& new_cslist, std::vector<int>& start_list) {
        new_cslist.clear();
        start_list.clear();
        start_list.push_back(0);
        for (int ctype = 1; ctype <= total_ctype; ctype++) {
            for (auto& ct : cslist) {
                if (ct->type == ctype) {
                    new_cslist.push_back(ct);
                }
            }
            int start = new_cslist.size();
            if (start != start_list.back()) start_list.push_back(start);
        }
    }
};

namespace ADU {

/// Built-in constraint type ids (must match the registered creators):
/// 1 = triangle stretch, 2 = tetrahedral strain, 3 = bending.
/// Defined after Mesh2Constraint so the creator lambdas can use its
/// curr_start_row bookkeeping. Lazily registered on first use.
inline void ensure_builtin_constraints_registered() {
    if (detail::builtin_registered()) return;
    detail::builtin_registered() = true;

    // type 1: triangle stretch
    register_constraint_creator(1, [](MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
                                      std::vector<std::shared_ptr<Constraint>>& constraints) {
#ifndef DISABLE_STRETCHING
        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);
        if (mesh.type == MeshData::TRI) {
            for (int fi = mesh.start_faceIdx; fi < mesh.end_faceIdx; fi++) {
                const auto& vinds = mesh_data.faces.row(fi);
                auto& ct = std::make_shared<TriangleConstraint>(
                    vinds, mesh_data.verts, material.tri_stretch_min, material.tri_stretch_max,
                    material.tri_stretch_k, material.tri_use_limit, constraints.size(), Mesh2Constraint::curr_start_row);
                Mesh2Constraint::curr_start_row += 2;
                constraints.emplace_back(ct);
            }
        }
#endif
    });

    // type 2: tetrahedral strain
    register_constraint_creator(2, [](MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
                                      std::vector<std::shared_ptr<Constraint>>& constraints) {
        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);
        if (mesh.type == MeshData::TET) {
            for (int ti = mesh.start_tetIdx; ti < mesh.end_tetIdx; ti++) {
                const auto& vinds = mesh_data.tets.row(ti);
                auto& ct = std::make_shared<TetrahedralConstraint>(
                    vinds, mesh_data.verts, material.tet_stretch_min, material.tet_stretch_max, material.tet_use_limit,
                    material.tet_stretch_k, constraints.size(), Mesh2Constraint::curr_start_row);
                Mesh2Constraint::curr_start_row += 3;
                constraints.emplace_back(ct);
            }
        }
    });

    // type 3: bending
    register_constraint_creator(3, [](MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
                                      std::vector<std::shared_ptr<Constraint>>& constraints) {
#ifndef DISABLE_BENDING
        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);
        if (mesh.type == MeshData::TRI) {
            for (int vi = mesh.start_vertIdx; vi < mesh.end_vertIdx; vi++) {
                const std::vector<int>& ring_idx = mesh_data.get_adjacency(vi);
                if (ring_idx.size() < 3) continue;
                auto& ct = std::make_shared<BendingConstraint>(
                    vi, ring_idx, mesh_data.verts, material.tri_bending_k,
                    constraints.size(), Mesh2Constraint::curr_start_row);
                Mesh2Constraint::curr_start_row += 1;
                constraints.emplace_back(ct);
            }
            printf("cur start row 2: %d\n", Mesh2Constraint::curr_start_row);
        }
#endif
    });
}

} // namespace ADU
