#pragma once

#include "solver/core/admm_backend.h"
#include "solver/backend_cpu/admm_impl_cpu.h"

#include <tbb/parallel_invoke.h>

namespace ADU {

/// CPU global solver: Eigen SimplicialLLT; the 3 RHS columns are solved in
/// parallel (Eigen's sparse triangular solve is not multithreaded itself).
class LinearSolverCPU : public ILinearSolver {
public:
    explicit LinearSolverCPU(ADMMImplCPU* impl) : impl_(impl) {}

    void solve() override {
        auto* s = impl_;
        const int nDynVert = s->static_vert_begin;
        tbb::parallel_invoke(
            [&]() { s->x_curr.block(0, 0, nDynVert, 1) = s->m_LLT_solver->solve(s->b.col(0)); },
            [&]() { s->x_curr.block(0, 1, nDynVert, 1) = s->m_LLT_solver->solve(s->b.col(1)); },
            [&]() { s->x_curr.block(0, 2, nDynVert, 1) = s->m_LLT_solver->solve(s->b.col(2)); });
    }

private:
    ADMMImplCPU* impl_;
};

} // namespace ADU
