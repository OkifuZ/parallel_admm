#pragma once

#include "mutils/common_types.h"

#include <array>
#include <cstddef>
#include <vector>

namespace ADU {

/// Host-facing collision-detection contract shared by the CPU detector
/// (ProximalQuery) and the GPU detector (BVH_GPU).
///
/// Device-side fast paths (e.g. BVH_GPU::update(Real* device_verts)) stay on
/// the concrete classes so device solvers can avoid host round-trips; this
/// interface covers everything a host solver or the application layer needs:
/// running detection from host positions, querying contact results, and
/// reading contact data for visualization.
class ICollisionDetector {
public:
    virtual ~ICollisionDetector() = default;

    /// Run collision detection (broad + narrow phase) against the given host positions.
    virtual void detect(const Matf_X3& pos) = 0;

    /// Number of contacts found by the last detect().
    virtual size_t contact_count() const = 0;

    /// Contact slots reserved at construction / init.
    virtual size_t max_slots() const = 0;

    /// Peak contact count seen so far (profiling / stats).
    virtual size_t peak_contact_count() const = 0;

    /// Contact points as host float3 array (visualization).
    /// NOTE: polyscope point clouds are registered with max_slots() points,
    /// so these must return arrays of exactly max_slots() entries
    /// (unused slots zero-filled); dynamic-size arrays break the GUI update.
    virtual std::vector<std::array<float, 3>> points() = 0;

    /// Contact normals as host float3 array (visualization). Same size
    /// contract as points().
    virtual std::vector<std::array<float, 3>> normals() = 0;

    /// Clear contact results (e.g. on reset).
    virtual void clear() = 0;
};

} // namespace ADU
