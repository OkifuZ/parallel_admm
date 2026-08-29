#pragma once

#include "mutils/common_types.h"
#include "solver/core/admm_config.h"

class Solver;

namespace ADU {

struct IContactSolver;

/// Abstract interface for ADMM backend (CPU or GPU).
/// Phase 1: minimal; Phase 3 will add linear_solver(), contact_solver(), etc.
struct IADMMBackend {
    virtual void precompute() = 0;
    virtual void step() = 0;
    virtual IContactSolver* contact_solver() { return nullptr; }
    virtual struct Solver* as_solver() = 0;
    virtual ~IADMMBackend() = default;

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

/// Placeholder for future: linear system solve (LLT vs Jacobi).
struct ILinearSolver {
    virtual ~ILinearSolver() = default;
};

/// Phase 2: unified contact subproblem (compute_Scc + project_feasible).
/// project_feasible: CPU uses (p, contacts, mu, max_iter); GPU ignores p/contacts, uses device state.
struct IContactSolver {
    virtual void compute_Scc() = 0;
    virtual void project_feasible(Matf_X3* p, void* contacts, Real mu, size_t max_iter) = 0;
    virtual ~IContactSolver() = default;
};

/// Placeholder for future: elastic constraint prox (D*x → z).
struct ILocalProjector {
    virtual ~ILocalProjector() = default;
};

} // namespace ADU
