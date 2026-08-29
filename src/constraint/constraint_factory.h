#pragma once

#include "mutils/common_types.h"
#include "mesh/mesh_container.h"
#include "mesh/mesh_phy_material.h"
#include "constraint/constraint.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

class MeshData;

namespace ADU {

/// Constraint creation callback: build the constraints of one mesh for one
/// type id and append them to `constraints`. `start_row` bookkeeping is done
/// by the callback via Mesh2Constraint::curr_start_row.
using ConstraintCreator = std::function<void(
    MeshData& mesh_data, size_t mesh_id, const PhyxMaterial& material,
    std::vector<std::shared_ptr<Constraint>>& constraints)>;

/// Register a creator for a constraint type id (1 = triangle stretch,
/// 2 = tetrahedral strain, 3 = bending, ...). New constraint types can be
/// plugged in without touching mesh_to_constraint.h.
void register_constraint_creator(int type_id, ConstraintCreator creator);

/// Invoke the registered creator for `type_id`; returns false when no
/// creator is registered for that type.
bool create_constraints_for_type(int type_id, MeshData& mesh_data, size_t mesh_id,
    const PhyxMaterial& material, std::vector<std::shared_ptr<Constraint>>& constraints);

/// Lazily register the built-in constraint creators (triangle / tet / bending).
void ensure_builtin_constraints_registered();

namespace detail {
inline std::unordered_map<int, ConstraintCreator>& constraint_creators() {
    static std::unordered_map<int, ConstraintCreator> registry;
    return registry;
}
inline bool& builtin_registered() {
    static bool registered = false;
    return registered;
}
} // namespace detail

inline void register_constraint_creator(int type_id, ConstraintCreator creator) {
    detail::constraint_creators()[type_id] = std::move(creator);
}

inline bool create_constraints_for_type(int type_id, MeshData& mesh_data, size_t mesh_id,
    const PhyxMaterial& material, std::vector<std::shared_ptr<Constraint>>& constraints) {
    ensure_builtin_constraints_registered();
    auto& registry = detail::constraint_creators();
    auto it = registry.find(type_id);
    if (it == registry.end()) return false;
    it->second(mesh_data, mesh_id, material, constraints);
    return true;
}

} // namespace ADU
