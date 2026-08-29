#pragma once

#include "solver/core/admm_backend.h"

namespace ADU {

class ADMMBackendGPU;

/// Phase 2: GPU contact solver, delegates to ADMMBackendGPU.
/// project_feasible ignores p/contacts; uses device state.
class ContactSolverGPU : public IContactSolver {
public:
    explicit ContactSolverGPU(ADMMBackendGPU* backend) : backend_(backend) {}

    void compute_Scc() override;
    void project_feasible(Matf_X3* p, void* contacts, Real mu, size_t max_iter) override;

private:
    ADMMBackendGPU* backend_;
};

} // namespace ADU
