#pragma once

#include "solver/solver.h"
#include "solver/xpbd_config.h"
#include "constraint/xpbd/xpbd_constraint.h"

namespace ADU {

/// Independent XPBD solver (Macklin et al. 2016), see
/// agent_aux/xpbd-integration-plan.md. Reuses the generic Solver state
/// (verts/vel/mass/pins/obstacles/animator/dt) and the app-injected
/// ICollisionDetector; it does NOT depend on ADMM internals.
class XPBDSolver : public Solver {
public:
    using XPBDConstraintList = std::vector<std::shared_ptr<XPBDConstraint>>;

    XPBDSolver();
    ~XPBDSolver();

    void init(const Matf_X3& verts, const Vecf_X& mass, Real dt) override;
    void step() override;
    void reset(const Matf_X3& ini_verts, bool need_precompute = false) override;

    /// Apply typed tuning parameters (from the [xpbd] TOML section).
    void apply_config(const XPBDConfig& cfg);

    // --- XPBD tuning (applied via apply_config; kept public for tests) ---
    unsigned int m_subSteps = 1;
    unsigned int m_maxIterations = 400;
    ADU::Real contact_stiffness = 10;
    bool use_GS_contact = true;

    void set_constraints(const XPBDConstraintList& cs) { m_xpbd_constraints = cs; }
    XPBDConstraintList& constraints() { return m_xpbd_constraints; }
    const XPBDConstraintList& constraints() const { return m_xpbd_constraints; }

private:
    XPBDConstraintList m_xpbd_constraints;
};

} // namespace ADU
