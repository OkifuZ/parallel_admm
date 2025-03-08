#include "constraint/constraint.h"
#include "constraint/bending_constraint.h"
#include "constraint/triangle_constraint.h"
#include "constraint/tetrahedral_constraint.h"
#include "constraint/pin_constraint.h"
#include "constraint/XPBD_constraints.h"
#include "constraint/nodal_collision_constraint.h"

#include "mesh/mesh_container.h"
#include "mesh/mesh_phy_material.h"

#include "mutils/common_types.h"

#include <vector>

//#define DISABLE_BENDING


class Mesh2Constraint {
public:
    static size_t curr_start_row;

    std::vector<int> invalid_inds;

    static void geometry_to_constraints(
        MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material, 
        std::vector<std::shared_ptr<Constraint>>& constraints, 
        std::vector<std::shared_ptr<XPBDConstraint>>& XPBD_constraints, int typeID)
    {
        using namespace ADU;

        MeshData::MeshObject& mesh = mesh_data.get_mesh(mesh_id);

        if (typeID == 1) {
            // tri
            if (mesh.type == MeshData::TRI) {
                for (int fi = mesh.start_faceIdx; fi < mesh.end_faceIdx; fi++) {
                    const auto& vinds = mesh_data.faces.row(fi);
                    auto& ct = std::make_shared<TriangleConstraint>(
                        vinds, mesh_data.verts, material.tri_stretch_min, material.tri_stretch_max,
                        material.tri_stretch_k, material.tri_use_limit, constraints.size(), curr_start_row);
                    curr_start_row += 2;
                    constraints.emplace_back(ct);

                    //auto xct = std::make_shared<FEMTriangleConstraint>();
                    //Real stiff = ct->k;
                    //bool res = xct->initConstraint(mesh_data.verts, vinds(0), vinds(1), vinds(2), 
                    //    stiff, stiff, stiff, 0.3_r, 0.3_r);
                    //ct->Binv_XPBD = xct->m_invRestMat;
                    //if (!res) ADU::make_exception("XPBD triangle constraint init issue");
                    ////xct->stiffness = 1e6;
                    //XPBD_constraints.push_back(xct);
                }
            }
        }
        if (typeID == 2) { // tet
            if (mesh.type == MeshData::TET) {
                // tet strain
                for (int ti = mesh.start_tetIdx; ti < mesh.end_tetIdx; ti++) {
                    const auto& vinds = mesh_data.tets.row(ti);
                    auto& ct = std::make_shared<TetrahedralConstraint>(
                        vinds, mesh_data.verts, material.tet_stretch_min, material.tet_stretch_max, material.tri_use_limit,
                        material.tet_stretch_k, constraints.size(), curr_start_row);
                    curr_start_row += 3;
                    constraints.emplace_back(ct);

                    /*auto xct = std::make_shared<XPBD_FEMTetConstraint>();
                    bool res = xct->initConstraint(mesh_data.verts, vinds(0), vinds(1), vinds(2), vinds(3), 2e2, 0.4_r);
                    if (!res) ADU::make_exception("XPBD FEM constraint init issue");
                    XPBD_constraints.push_back(xct);*/
                }
            }
        }

        if (typeID == 3) {
            if (mesh.type == MeshData::TRI) { // streching and bending
#ifndef DISABLE_BENDING
                // bending
                for (int vi = mesh.start_vertIdx; vi < mesh.end_vertIdx; vi++) {
                    const std::vector<int>& ring_idx = mesh_data.get_adjacency(vi);
                    if (ring_idx.size() < 3) continue;
                    auto& ct = std::make_shared<BendingConstraint>(
                        vi, ring_idx, mesh_data.verts, material.tri_bending_k,
                        constraints.size(), curr_start_row);
                    curr_start_row += 1;
                    constraints.emplace_back(ct);
                    //if (mesh_data.on_boundary(vi)) {
                    //    
                    //} // boundary points, ignore
                    //else {
                    //    const std::vector<int>& ring_idx = mesh_data.get_adjacency(vi);
                    //    auto& ct = std::make_shared<BendingConstraint>(
                    //        vi, ring_idx, mesh_data.verts, material.tri_bending_k,
                    //        constraints.size(), curr_start_row);
                    //    curr_start_row += 1;
                    //    constraints.emplace_back(ct);
                    //}

                    /*auto xct = std::make_shared<XPBDBendingConstraint>();
                    bool res = xct->initConstraint(ct);
                    if (!res) ADU::make_exception("XPBD bending constraint init issue");
                    xct->stiffness = 0.001_r;
                    XPBD_constraints.push_back(xct);*/
                }
                printf("cur start row 2: %d\n", curr_start_row);
#endif
            }
        }

        //if (mesh.type == MeshData::TRI) { // streching and bending
        //    // stretching
        //    for (int fi = mesh.start_faceIdx; fi < mesh.end_faceIdx; fi++) {
        //        const auto& vinds = mesh_data.faces.row(fi);
        //        auto& ct = std::make_shared<TriangleConstraint>(
        //            vinds, mesh_data.verts, material.tri_stretch_min, material.tri_stretch_max, 
        //            material.tri_stretch_k, material.tri_use_limit, constraints.size(), curr_start_row);
        //        curr_start_row += 2;
        //        constraints.emplace_back(ct);

        //        //auto xct = std::make_shared<FEMTriangleConstraint>();
        //        //Real stiff = ct->k;
        //        //bool res = xct->initConstraint(mesh_data.verts, vinds(0), vinds(1), vinds(2), 
        //        //    stiff, stiff, stiff, 0.3_r, 0.3_r);
        //        //ct->Binv_XPBD = xct->m_invRestMat;
        //        //if (!res) ADU::make_exception("XPBD triangle constraint init issue");
        //        ////xct->stiffness = 1e6;
        //        //XPBD_constraints.push_back(xct);
        //    }

        //    printf("\ncur start row 1: %d\n", curr_start_row);

        //    #ifndef DISABLE_BENDING
        //    // bending
        //    for (int vi = mesh.start_vertIdx; vi < mesh.end_vertIdx; vi++) {
        //        const std::vector<int>& ring_idx = mesh_data.get_adjacency(vi);
        //        if (ring_idx.size() < 3) continue;
        //        auto& ct = std::make_shared<BendingConstraint>(
        //            vi, ring_idx, mesh_data.verts, material.tri_bending_k,
        //            constraints.size(), curr_start_row);
        //        curr_start_row += 1;
        //        constraints.emplace_back(ct);
        //        //if (mesh_data.on_boundary(vi)) {
        //        //    
        //        //} // boundary points, ignore
        //        //else {
        //        //    const std::vector<int>& ring_idx = mesh_data.get_adjacency(vi);
        //        //    auto& ct = std::make_shared<BendingConstraint>(
        //        //        vi, ring_idx, mesh_data.verts, material.tri_bending_k,
        //        //        constraints.size(), curr_start_row);
        //        //    curr_start_row += 1;
        //        //    constraints.emplace_back(ct);
        //        //}

        //        /*auto xct = std::make_shared<XPBDBendingConstraint>();
        //        bool res = xct->initConstraint(ct);
        //        if (!res) ADU::make_exception("XPBD bending constraint init issue");
        //        xct->stiffness = 0.001_r;
        //        XPBD_constraints.push_back(xct);*/
        //    }

        //    printf("cur start row 2: %d\n", curr_start_row);

        //    #endif
        //}
        //else if (mesh.type == MeshData::TET) {
        //    // tet strain
        //    for (int ti = mesh.start_tetIdx; ti < mesh.end_tetIdx; ti++) {
        //        const auto& vinds = mesh_data.tets.row(ti);
        //        auto& ct = std::make_shared<TetrahedralConstraint>(
        //            vinds, mesh_data.verts, material.tet_stretch_min, material.tet_stretch_max, material.tri_use_limit,
        //            material.tet_stretch_k,  constraints.size(), curr_start_row);
        //        curr_start_row += 3;
        //        constraints.emplace_back(ct);

        //        auto xct = std::make_shared<XPBD_FEMTetConstraint>();
        //        bool res = xct->initConstraint(mesh_data.verts, vinds(0), vinds(1), vinds(2), vinds(3), 2e2, 0.4_r);
        //        if (!res) ADU::make_exception("XPBD FEM constraint init issue");
        //        XPBD_constraints.push_back(xct);
        //    }
        //}
        //else {

        //}
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

size_t Mesh2Constraint::curr_start_row = 0;

