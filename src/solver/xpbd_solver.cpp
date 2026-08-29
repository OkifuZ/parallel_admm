#include "solver/xpbd_solver.h"
#include "constraint/xpbd/mesh_to_xpbd_constraint.h"
#include "mutils/exception_handle.h"

#include <chrono>

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
    enable_frictional_contact = cfg.enable_frictional_contact;
    dcd_interval = cfg.dcd_interval;
    use_unique_contact = cfg.use_unique_contact;
    mu = cfg.mu;
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
    using namespace ADU;
    const auto t0 = std::chrono::steady_clock::now();
    const Real h = m_dt / static_cast<Real>(m_subSteps);
    const Real h_inv = 1.0_r / h;

    // Pin vertices stay fixed: gravity/integration skips them and their
    // velocity is kept at zero, so x does not drift.
    Matf_X3 x_prev(m_vertices);
    Matf_X3 x_curr(m_vertices);
    Matf_X3 v_curr(m_velocities);

    for (unsigned int substep = 0; substep < m_subSteps; substep++) {
        x_prev = x_curr;
        for (int i = 0; i < m_nVert; i++) {
            if (m_pin_inds_set.count(i)) continue;
            v_curr.row(i).y() -= g * h;
        }
        x_curr += v_curr * h;

        for (unsigned int iter = 0; iter < m_maxIterations; iter++) {
            // collision detection (broad + narrow phase, host positions)
            if (enable_frictional_contact && prox_query && (iter % dcd_interval == 0)) {
                prox_query->proximal_query(x_curr);
                if (use_unique_contact) prox_query->unique_contact();
            }

            // elastic constraints (FEM triangle / tet / bending)
            for (auto& ct : m_xpbd_constraints) {
                ct->updateConstraint();
                ct->solvePositionConstraint(x_curr, Matf_X3{}, m_M_inv_vec, iter, h);
            }

            // frictional contacts as PT/EE XPBD distance constraints
            // (compression stiffness = contact_stiffness; 5 Gauss-Seidel passes
            // per iteration, matching the pre-archive behavior).
            if (enable_frictional_contact && prox_query) {
                for (int pass = 0; pass < 5; pass++) {
                    solve_contact_constraints(prox_query->contact_info_list, m_M_inv_vec, x_curr);
                }
            }
        }

        v_curr = h_inv * (x_curr - x_prev);
        v_curr *= 0.9995_r;
    }
    m_velocities = v_curr;
    m_vertices = x_curr;
    step_cnt++;
    const double step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    step_metrics.push_back({ step_cnt, step_ms, prox_query ? prox_query->contact_info_list.size() : 0 });
}

void XPBDSolver::solve_contact_constraints(const ProximalQuery::ContactInfoList& contacts,
    const ADU::Vecf_X& M_inv_list, ADU::Matf_X3& pos) {
    using namespace ADU;
    std::array<Vecf_3, 4> corr;
    const Real compress_stiff = contact_stiffness;
    const Real alpha = 1.0_r;

    for (size_t i = 0; i < contacts.size(); i++) {
        const auto& ct = contacts[i];
        const auto& inds = ct.vinds;

        const Real m_inv0 = M_inv_list[inds[0]];
        const Real m_inv1 = M_inv_list[inds[1]];
        const Real m_inv2 = M_inv_list[inds[2]];
        const Real m_inv3 = M_inv_list[inds[3]];

        bool res{ false };
        if (ct.pair_type == ContactInfo::PT) {
            res = XPBD_utils::solve_TrianglePointDistanceConstraint(
                pos.row(inds[0]), m_inv0,
                pos.row(inds[1]), m_inv1,
                pos.row(inds[2]), m_inv2,
                pos.row(inds[3]), m_inv3,
                ContactParameter::get_thickness(),
                compress_stiff, 0,
                corr[0], corr[1], corr[2], corr[3]);
        } else if (ct.pair_type == ContactInfo::EE) {
            res = XPBD_utils::solve_EdgeEdgeDistanceConstraint(
                pos.row(inds[0]), m_inv0,
                pos.row(inds[1]), m_inv1,
                pos.row(inds[2]), m_inv2,
                pos.row(inds[3]), m_inv3,
                ContactParameter::get_thickness(),
                compress_stiff, 0,
                corr[0], corr[1], corr[2], corr[3]);
        }

        if (res) {
            pos.row(inds[0]) += corr[0] * alpha;
            pos.row(inds[1]) += corr[1] * alpha;
            pos.row(inds[2]) += corr[2] * alpha;
            pos.row(inds[3]) += corr[3] * alpha;
        }
    }
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
