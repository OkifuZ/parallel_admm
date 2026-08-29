#pragma once

#include "solver/solver.h"
#include "solver/core/admm_backend.h"

namespace ADU {

/// Phase 3: ADMMSolver holds IADMMBackend, delegates step/init/reset to backend.
/// Use inner_solver() for setup (add_constraints, prox_query, etc.).
class ADMMSolver : public Solver {
public:
    explicit ADMMSolver(std::unique_ptr<IADMMBackend> backend) : backend_(std::move(backend)) {}

    Solver* inner_solver() { return backend_ ? backend_->as_solver() : nullptr; }
    const Solver* inner_solver() const { return backend_ ? backend_->as_solver() : nullptr; }

    void init(const Matf_X3& verts, const Vecf_X& mass, Real dt) override;
    void step() override;
    void reset(const Matf_X3& ini_verts, bool need_precompute = false) override;

    const Matf_X3& getVertices() const override;
    Matf_X3& getVertices() override;
    const Matf_X3& getVelocities() const override;
    Matf_X3& getVelocities() override;

private:
    std::unique_ptr<IADMMBackend> backend_;
};

} // namespace ADU
