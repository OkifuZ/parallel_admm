#pragma once

#include "mutils/common_types.h"

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
};

/// Placeholder for future: linear system solve (LLT vs Jacobi).
struct ILinearSolver {
    virtual ~ILinearSolver() = default;
};

/// Phase 2: unified contact subproblem (compute_Scc + project_feasible).
/// project_feasible: CPU uses (p, contacts, mu, max_iter); GPU ignores p/contacts, uses device state.
struct IContactSolver {
    virtual void compute_Scc(bool is_XPBD = false) = 0;
    virtual void project_feasible(Matf_X3* p, void* contacts, Real mu, size_t max_iter) = 0;
    virtual ~IContactSolver() = default;
};

/// Placeholder for future: elastic constraint prox (D*x → z).
struct ILocalProjector {
    virtual ~ILocalProjector() = default;
};

} // namespace ADU
