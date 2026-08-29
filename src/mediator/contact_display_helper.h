#pragma once

#include "contact/icollision_detector.h"

namespace polyscope {
class PointCloud;
}

class Solver;

namespace ADU {

/// Update contact point cloud display through the ICollisionDetector interface.
/// `solver` is optional and only used for the ADMM-specific "S" vector quantity
/// (CPU contact info); pass nullptr to skip it.
void update_contact_display(ICollisionDetector* detector, Solver* solver, polyscope::PointCloud* vis_contactP);

} // namespace ADU
