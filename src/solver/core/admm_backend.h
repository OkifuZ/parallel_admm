#pragma once

#include "mutils/common_types.h"
#include "solver/core/admm_config.h"

class Solver;

namespace ADU {

struct IContactSolver;
struct ILinearSolver;
struct ILocalProjector;

/// Abstract interface for ADMM backend (CPU or GPU).
struct IADMMBackend {
    virtual void precompute() = 0;
    virtual void step() = 0;
    virtual IContactSolver* contact_solver() { return nullptr; }
    virtual struct Solver* as_solver() = 0;
    virtual ~IADMMBackend() = default;

    /// Sub-problem components (see below). May return nullptr on backends
    /// that keep these steps inline.
    virtual ILinearSolver* linear_solver() { return nullptr; }
    virtual ILocalProjector* local_projector() { return nullptr; }

    /// Apply tuning parameters derived from the scene config. No constraint
    /// dependency; must be called before init().
    virtual void apply_config(const ADMMSolverConfig& cfg) = 0;

    /// Called by the app layer AFTER all constraints are registered:
    /// GPU backend converts host constraints to device here; CPU is a no-op.
    virtual void finalize_constraints() = 0;

    /// ADMM constraint dimension (m_nCDim), known only after constraint assembly.
    virtual void set_constraint_dim(int n) = 0;

    /// Toggle per-step host <-> device sync (GPU backend). Default true;
    /// set false for headless runs without visualization/export/animators.
    virtual void set_sync_to_host(bool /*enabled*/) {}
};

/// Global linear system solve (the "global step": A x = b).
/// CPU: Eigen SimplicialLLT, the 3 RHS columns solved in parallel.
/// GPU: iterative Jacobi on device state.
struct ILinearSolver {
    virtual ~ILinearSolver() = default;
    /// Solve the current global system into the solver state (x_curr).
    virtual void solve() = 0;
};

/// Elastic constraint prox (the "local step": z = prox(D*x + Ue)).
struct ILocalProjector {
    virtual ~ILocalProjector() = default;
    /// Compute z = prox(D*x + Ue) into the solver state.
    virtual void project_elastic() = 0;
};

/// Phase 2: unified contact subproblem (compute_Scc + project_feasible).
/// project_feasible: CPU uses (p, contacts, mu, max_iter); GPU ignores p/contacts, uses device state.
struct IContactSolver {
    virtual void compute_Scc() = 0;
    virtual void project_feasible(Matf_X3* p, void* contacts, Real mu, size_t max_iter) = 0;
    virtual ~IContactSolver() = default;
};

} // namespace ADU
