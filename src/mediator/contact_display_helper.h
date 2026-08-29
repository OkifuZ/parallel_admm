#pragma once

namespace polyscope {
class PointCloud;
}

namespace ADU {

class ADMMSolver;

/// Update contact point cloud display. Hides impl cast; application layer needs no impl headers.
void update_contact_display(ADMMSolver* admm_solver, bool use_gpu, polyscope::PointCloud* vis_contactP);

} // namespace ADU
