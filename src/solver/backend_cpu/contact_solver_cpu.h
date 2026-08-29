#pragma once

#include "solver/core/admm_backend.h"
#include "solver/backend_cpu/admm_impl_cpu.h"
#include "contact/narrow_phase.h"

namespace ADU {

/// Phase 2: CPU contact solver, delegates to ADMMImplCPU.
class ContactSolverCPU : public IContactSolver {
public:
    explicit ContactSolverCPU(ADMMImplCPU* impl) : impl_(impl) {}

    void compute_Scc(bool is_XPBD = false) override { impl_->compute_Scc(is_XPBD); }
    void project_feasible(Matf_X3* p, void* contacts, Real mu, size_t max_iter) override;

private:
    ADMMImplCPU* impl_;
};

} // namespace ADU
