#include "solver/backend_gpu/contact_solver_gpu.h"
#include "solver/backend_gpu/admm_backend_gpu.h"

namespace ADU {

void ContactSolverGPU::compute_Scc(bool is_XPBD) {
    backend_->ContactSolverGPU_do_compute_Scc(is_XPBD);
}

void ContactSolverGPU::project_feasible(Matf_X3* /*p*/, void* /*contacts*/, Real /*mu*/, size_t /*max_iter*/) {
    backend_->ContactSolverGPU_do_project_feasible();
}

} // namespace ADU
