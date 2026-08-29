#pragma once

/// Phase 3.4: CPU backend using composition. Holds ADMMImplCPU, implements IADMMBackend.

#include <memory>
#include "solver/core/admm_backend.h"
#include "solver/backend_cpu/contact_solver_cpu.h"
#include "solver/backend_cpu/admm_impl_cpu.h"
#include "solver/solver.h"

namespace ADU {

class ADMMBackendCPU : public Solver, public IADMMBackend {
public:
    ADMMBackendCPU() : impl_(std::make_unique<ADMMImplCPU>()), contact_solver_(std::make_unique<ContactSolverCPU>(impl_.get())) {}

    void precompute() override { impl_->precompute(); }
    void step() override { impl_->step(); }
    IContactSolver* contact_solver() override { return contact_solver_.get(); }
    Solver* as_solver() override { return impl_.get(); }

    ADMMImplCPU* impl() { return impl_.get(); }
    const ADMMImplCPU* impl() const { return impl_.get(); }

private:
    std::unique_ptr<ADMMImplCPU> impl_;
    std::unique_ptr<ContactSolverCPU> contact_solver_;
};

} // namespace ADU
