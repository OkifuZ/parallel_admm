#include "solver/backend_cpu/contact_solver_cpu.h"

namespace ADU {

void ContactSolverCPU::project_feasible(Matf_X3* p, void* contacts, Real mu, size_t max_iter) {
    if (!p || !contacts) return;
    auto& cl = *static_cast<ProximalQuery::ContactInfoList*>(contacts);
    impl_->project_feasible(*p, cl, mu, max_iter);
}

} // namespace ADU
