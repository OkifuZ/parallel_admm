#include "mediator/contact_display_helper.h"
#include "solver/core/admm_solver.h"
#include "solver/backend_cpu/admm_impl_cpu.h"
#include "solver/backend_gpu/admm_impl_gpu.h"
#include "polyscope/point_cloud.h"

namespace ADU {

void update_contact_display(ADMMSolver* admm_solver, bool use_gpu, polyscope::PointCloud* vis_contactP) {
    if (!admm_solver || !vis_contactP) return;
    Solver* inner = admm_solver->inner_solver();
    if (!inner || !inner->prox_query) return;

    auto* impl_cpu = dynamic_cast<ADMMImplCPU*>(inner);
    if (!impl_cpu) return;

    auto& s_vec = impl_cpu->prox_query->get_SaSb_mean();
    vis_contactP->addVectorQuantity("S", s_vec);

    if (!use_gpu) {
        vis_contactP->updatePointPositions(inner->prox_query->get_contact_points());
        vis_contactP->addVectorQuantity("normal", inner->prox_query->get_contact_normals());
    } else {
        auto* impl_gpu = dynamic_cast<ADMMImplGPU*>(inner);
        if (impl_gpu) {
            vis_contactP->updatePointPositions(impl_gpu->getContactPoints());
            vis_contactP->addVectorQuantity("normal", impl_gpu->getContactNormals());
        }
    }
}

} // namespace ADU
