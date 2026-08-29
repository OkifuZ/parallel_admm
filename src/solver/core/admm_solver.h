#pragma once

#include "solver/solver.h"
#include "solver/core/admm_backend.h"

namespace ADU {

/// Phase 3: ADMMSolver holds IADMMBackend, delegates step/init/reset to backend.
/// Use inner_solver() for setup (add_constraints, prox_query, etc.).
/// All generic Solver setup calls are forwarded to the inner solver so the
/// facade itself is safe to use directly (no accidental state split).
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

    // --- typed config entry points (no dynamic_cast at the app layer) ---
    void apply_config(const ADMMSolverConfig& cfg) { if (backend_) backend_->apply_config(cfg); }
    void finalize_constraints() { if (backend_) backend_->finalize_constraints(); }
    void set_constraint_dim(int n) { if (backend_) backend_->set_constraint_dim(n); }
    void set_sync_to_host(bool enabled) { if (backend_) backend_->set_sync_to_host(enabled); }

    // --- generic setup forwarded to the inner solver ---
    void addPins(const std::vector<int>& pin_inds) override;
    void add_constraints(const ConstraintsList& constraints_) override;
    void add_nodal_constraints(const NodalCollisionConstraintsList& constraints_) override;
    void addObstacle(const std::shared_ptr<Obstacle>& obstacle) override;

private:
    std::unique_ptr<IADMMBackend> backend_;
};

} // namespace ADU
