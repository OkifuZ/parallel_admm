#pragma once

#include "mutils/common_types.h"
#include "mesh/mesh_container.h"
#include "mesh/mesh_phy_material.h"
#include "constraint/xpbd/xpbd_constraint.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ADU {

/// XPBD constraint creation callback: build the XPBD constraints of one mesh
/// for one type id and append them to `constraints`.
using XPBDConstraintCreator = std::function<void(
    MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
    std::vector<std::shared_ptr<XPBDConstraint>>& constraints)>;

/// Register a creator for an XPBD constraint type id
/// (1 = FEM triangle stretch, 2 = FEM tet strain, 3 = bending).
void register_xpbd_constraint_creator(int type_id, XPBDConstraintCreator creator);

/// Invoke the registered creator for `type_id`; returns false when no
/// creator is registered for that type.
bool create_xpbd_constraints_for_type(int type_id, MeshData& mesh_data, size_t mesh_id,
    const PhyxMaterial& material, std::vector<std::shared_ptr<XPBDConstraint>>& constraints);

/// Lazily register the built-in XPBD constraint creators
/// (definition lives in mesh_to_xpbd_constraint.h).
void ensure_builtin_xpbd_constraints_registered();

namespace xpbd_detail {
inline std::unordered_map<int, XPBDConstraintCreator>& xpbd_creators() {
    static std::unordered_map<int, XPBDConstraintCreator> registry;
    return registry;
}
inline bool& xpbd_builtin_registered() {
    static bool registered = false;
    return registered;
}
} // namespace xpbd_detail

inline void register_xpbd_constraint_creator(int type_id, XPBDConstraintCreator creator) {
    xpbd_detail::xpbd_creators()[type_id] = std::move(creator);
}

inline bool create_xpbd_constraints_for_type(int type_id, MeshData& mesh_data, size_t mesh_id,
    const PhyxMaterial& material, std::vector<std::shared_ptr<XPBDConstraint>>& constraints) {
    ensure_builtin_xpbd_constraints_registered();
    auto& registry = xpbd_detail::xpbd_creators();
    auto it = registry.find(type_id);
    if (it == registry.end()) return false;
    it->second(mesh_data, mesh_id, material, constraints);
    return true;
}

} // namespace ADU
