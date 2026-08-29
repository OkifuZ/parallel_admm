#include "solver/xpbd_solver.h"
#include "constraint/xpbd/mesh_to_xpbd_constraint.h"
#include "mutils/exception_handle.h"

namespace ADU {

XPBDSolver::XPBDSolver() {
    // Compile anchor: pulls in the XPBD constraint module and lazily
    // registers the built-in constraint creators.
    ensure_builtin_xpbd_constraints_registered();
}

XPBDSolver::~XPBDSolver() = default;

void XPBDSolver::apply_config(const XPBDConfig& cfg) {
    m_subSteps = cfg.substeps;
    m_maxIterations = cfg.max_iter;
    contact_stiffness = cfg.contact_stiffness;
    use_GS_contact = cfg.use_GS_contact;
    g = cfg.g;
    // P4 will wire: enable_frictional_contact / use_CCD / mu
}

void XPBDSolver::init(const Matf_X3& verts, const Vecf_X& mass, Real dt) {
    // Mirrors Solver::init but without the ADMM constraint-container check:
    // XPBD keeps its own constraint list (m_xpbd_constraints).
    m_vertices = verts;
    m_velocities.resizeLike(m_vertices);
    m_velocities.setZero();
    m_M_vec = mass;
    m_M_inv_vec = mass;
    for (int i = 0; i < m_M_inv_vec.rows(); i++) m_M_inv_vec(i) = 1.0_r / m_M_vec(i);

    m_dt = dt;
    m_dt2 = m_dt * m_dt;
    m_dt_inv = 1.0_r / m_dt;
    m_nVert = verts.rows();
}

void XPBDSolver::step() {
    // P2 skeleton: gravity + explicit position update.
    // Elastic/contact constraints are wired in P3/P4.
    const Real h = m_dt / static_cast<Real>(m_subSteps);
    for (int i = 0; i < m_nVert; i++) {
        if (m_pin_inds_set.count(i)) continue;
        m_velocities.row(i).y() -= g * h;
    }
    m_vertices += m_velocities * h;
}

void XPBDSolver::reset(const Matf_X3& ini_verts, bool need_precompute) {
    m_vertices = ini_verts;
    m_velocities.setZero();
    if (animator) {
        animator->reset();
        animator->animate_all(m_vertices, m_vertices, m_velocities, m_dt);
    }
}

} // namespace ADU
