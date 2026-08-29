#include "contact/contact_display.h"
#include "solver/solver.h"
#include "polyscope/point_cloud.h"

namespace ADU {

void update_contact_display(ICollisionDetector* detector, Solver* solver, polyscope::PointCloud* vis_contactP) {
    if (!detector || !vis_contactP) return;

    vis_contactP->updatePointPositions(detector->points());
    vis_contactP->addVectorQuantity("normal", detector->normals());

    // "S" vector quantity: ADMM-specific, sourced from the CPU contact info
    // (kept for backward compatibility with the old display).
    if (solver && solver->prox_query) {
        auto& s_vec = solver->prox_query->get_SaSb_mean();
        vis_contactP->addVectorQuantity("S", s_vec);
    }
}

} // namespace ADU
