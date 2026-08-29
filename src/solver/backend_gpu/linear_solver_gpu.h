#pragma once

#include "solver/core/admm_backend.h"
#include "solver/backend_gpu/admm_impl_gpu.h"

namespace ADU {

/// GPU global solver: iterative Jacobi on device state.
class LinearSolverGPU : public ILinearSolver {
public:
    explicit LinearSolverGPU(ADMMImplGPU* impl) : impl_(impl) {}

    void solve() override { impl_->do_jacobi_global(); }

private:
    ADMMImplGPU* impl_;
};

} // namespace ADU
