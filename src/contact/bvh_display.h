#pragma once

#include <array>
#include <vector>

#include <glm/vec3.hpp>

#include "contact/collision_detection_cuda.h"

namespace ADU {

/// Convert BVH_GPU boxes to wireframe nodes and edges for polyscope CurveNetwork.
inline void update_bvh_geometry(const BVH_GPU* bvh,
    std::vector<glm::vec3>& nodes,
    std::vector<std::array<size_t, 2>>& edges)
{
    if (!bvh) return;

    nodes.clear();
    edges.clear();

    for (size_t j = 0; j < bvh->bvs.size(); j++) {
        float ox = bvh->bvs[j].lower.x;
        float oy = bvh->bvs[j].lower.y;
        float oz = bvh->bvs[j].lower.z;
        float bwidth = bvh->bvs[j].upper.x - bvh->bvs[j].lower.x;
        float bheight = bvh->bvs[j].upper.y - bvh->bvs[j].lower.y;
        float blength = bvh->bvs[j].upper.z - bvh->bvs[j].lower.z;

        size_t start = nodes.size();
        nodes.push_back(glm::vec3{ ox, oy, oz });
        nodes.push_back(glm::vec3{ ox, oy, oz + blength });
        nodes.push_back(glm::vec3{ ox, oy + bheight, oz });
        nodes.push_back(glm::vec3{ ox + bwidth, oy, oz });
        nodes.push_back(glm::vec3{ ox + bwidth, oy + bheight, oz });
        nodes.push_back(glm::vec3{ ox + bwidth, oy, oz + blength });
        nodes.push_back(glm::vec3{ ox, oy + bheight, oz + blength });
        nodes.push_back(glm::vec3{ ox + bwidth, oy + bheight, oz + blength });

        edges.push_back({ start + 0, start + 1 });
        edges.push_back({ start + 0, start + 2 });
        edges.push_back({ start + 0, start + 3 });
        edges.push_back({ start + 2, start + 4 });
        edges.push_back({ start + 2, start + 6 });
        edges.push_back({ start + 3, start + 4 });
        edges.push_back({ start + 3, start + 5 });
        edges.push_back({ start + 4, start + 7 });
        edges.push_back({ start + 5, start + 1 });
        edges.push_back({ start + 5, start + 7 });
        edges.push_back({ start + 6, start + 1 });
        edges.push_back({ start + 6, start + 7 });
    }
}

} // namespace ADU
