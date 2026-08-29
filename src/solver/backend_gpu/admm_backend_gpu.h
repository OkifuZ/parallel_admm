#pragma once

/// Phase 3.4: GPU backend using composition. Holds ADMMImplGPU, implements IADMMBackend.

#include <memory>
#include "solver/core/admm_backend.h"
#include "solver/backend_gpu/contact_solver_gpu.h"
#include "solver/backend_gpu/admm_impl_gpu.h"
#include "solver/solver.h"

namespace ADU {

class ADMMBackendGPU : public Solver, public IADMMBackend {
public:
    ADMMBackendGPU() : impl_(std::make_unique<ADMMImplGPU>()), contact_solver_(std::make_unique<ContactSolverGPU>(this)) {}

    void precompute() override { impl_->precompute(); }
    void step() override { impl_->step(); }
    IContactSolver* contact_solver() override { return contact_solver_.get(); }
    ILinearSolver* linear_solver() override { return impl_->linear_solver(); }
    ILocalProjector* local_projector() override { return impl_->local_projector(); }
    Solver* as_solver() override { return impl_.get(); }

    void apply_config(const ADMMSolverConfig& cfg) override {
        impl_->apply_config(cfg);
        impl_->Global_Jacobi_iter = cfg.global_jacobi_iter;
    }
    void finalize_constraints() override { impl_->convert_constraint2device(); }
    void set_constraint_dim(int n) override { impl_->m_nCDim = n; }
    void set_sync_to_host(bool enabled) override { impl_->set_sync_to_host(enabled); }

    ADMMImplGPU* impl() { return impl_.get(); }
    const ADMMImplGPU* impl() const { return impl_.get(); }

    void ContactSolverGPU_do_compute_Scc() { impl_->do_compute_Scc(); }
    void ContactSolverGPU_do_project_feasible() { impl_->do_project_feasible_parallel(); }

private:
    std::unique_ptr<ADMMImplGPU> impl_;
    std::unique_ptr<ContactSolverGPU> contact_solver_;
};

} // namespace ADU
