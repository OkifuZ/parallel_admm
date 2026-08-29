/// Phase 3.4: CPU ADMM implementation (migrated from admm_full_solver.cpp + admm_full_solver_RL_damping.cpp).

#include "solver/backend_cpu/admm_impl_cpu.h"
#include "mutils/timer.h"
#include "constraint/pin_constraint.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>
#include <mutils/common_type_hostonly.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <mutex>
#include <memory>
#include <random>
#include <chrono>

namespace ADU {

void ADMMImplCPU::apply_config(const ADMMSolverConfig& cfg) {
    static_mesh_id_begin = cfg.static_mesh_id_begin;
    static_vert_begin = cfg.static_vert_begin;
    vinds_surf_set = cfg.vinds_surf_set;

    admm_max_iter = cfg.admm_max_iter;
    collision_detection_interval = cfg.DCD_interval;
    gs_max_iter = cfg.gs_max_iter;
    warmstart_Ue = cfg.warmstart_Ue;
    warmstart_Uc = cfg.warmstart_Uc;
    use_jacobi = cfg.use_jacobi;
    g = cfg.g;

    enable_frictional_contact = cfg.enable_frictional_contact;
    use_CCD = cfg.use_CCD;
    use_unique_contact = cfg.use_unique_contact;
    mu = cfg.mu;
    kappa = cfg.kappa;
    beta = cfg.beta;
    use_heuristic_W_c = cfg.use_heuristic_wc;
    heu_sigma = cfg.wc_sigma;
    heu_beta = cfg.wc_beta;
    contact_w_list = cfg.contact_w_list;
    contact_w_inv_list = cfg.contact_w_inv_list;

    damp_k_L = cfg.damp_k_L;
    damp_k_M = cfg.damp_k_M;
}

void ADMMImplCPUBase::precompute() {
    make_exception("ADMMImplCPUBase::precompute() deprecated");
}

void ADMMImplCPUBase::init() {
    Timer timer("ADMMImplCPUBase::init()");
    this->precompute();
}

void ADMMImplCPUBase::reset(const ADU::Matf_X3& ini_verts, bool need_precompute) {
    this->m_vertices = ini_verts;
    this->m_velocities.setZero();
    if (warmstart_Ue) m_Ue.setZero();
    if (warmstart_Uc) m_Uc.setZero();
    if (need_precompute) precompute();
    if (coloring_parallel_contact) {
        v_c_list.resize(m_vertices.rows());
        for (size_t i = 0; i < v_c_list.size(); i++) v_c_list[i].clear();
        color_result.resize(_color_max_num + 1);
        for (int i = 0; i < _color_max_num; i++) color_result[i].clear();
    }
    if (prox_query) prox_query->contact_info_list.clear();
    if (animator) {
        animator->reset();
        animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
    }
}

void ADMMImplCPUBase::project_feasible(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_GS_iter) {
    (void)p;
    (void)contacts;
    (void)mu;
    (void)max_GS_iter;
    make_exception("use ADMMImplCPU::project_feasible");
}

void ADMMImplCPUBase::sequential_greedy_coloring(const ProximalQuery::ContactInfoList& cs_list, bool use_rand) {
    for (int i = 0; i <= _color_max_num; i++) color_result[i].clear();
    for (auto& csl : v_c_list) csl.clear();
    for (size_t ci = 0; ci < cs_list.size(); ci++) {
        const auto& c = cs_list[ci];
        v_c_list[c.vinds[0]].push_back(ci);
        v_c_list[c.vinds[1]].push_back(ci);
        v_c_list[c.vinds[2]].push_back(ci);
        v_c_list[c.vinds[3]].push_back(ci);
    }
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, _color_max_num);
    ct_colors = std::vector<int>(static_cast<int>(cs_list.size()), 0);
    ADU::Veci_X neighbor_color(_color_max_num + 1);
    for (size_t ci = 0; ci < cs_list.size(); ci++) {
        neighbor_color.setZero();
        const auto& c = cs_list[ci];
        for (size_t vi : c.vinds) {
            for (size_t nci : v_c_list[vi]) neighbor_color(ct_colors[nci]) = 1;
        }
        int start_color = use_rand ? uni(rng) : 0;
        bool find_one = false;
        for (int color = start_color; color < start_color + _color_max_num; color++) {
            int real_c = color % _color_max_num + 1;
            if (neighbor_color(real_c) == 0) {
                ct_colors[ci] = real_c;
                color_result[real_c].push_back(static_cast<size_t>(ci));
                find_one = true;
                break;
            }
        }
        if (!find_one) color_result[0].push_back(ci);
    }
}

void ADMMImplCPU::init() {
    Timer timer("ADMMImplCPU::init()");
    this->precompute();
    for (size_t ni = m_nodal_collision_constraint_start; ni < m_nodal_collision_constraint_end; ni++) {
        auto ct = std::static_pointer_cast<NodalCollisionConstraint>(m_constraints[ni]);
        ct->contact_info = nullptr;
        ct->active = false;
    }
    if (coloring_parallel_contact) {
        v_c_list.resize(m_vertices.rows());
        for (size_t i = 0; i < v_c_list.size(); i++) v_c_list[i].reserve(64);
        color_result.resize(_color_max_num + 1);
        for (int i = 0; i < _color_max_num; i++) color_result[i].reserve(2000);
    }
    if (animator) {
        animator->reset();
        animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
    }
    printf("ADMMImplCPU init done\n");
}

void ADMMImplCPU::precompute() {
    using namespace ADU;
    printf("ADMMImplCPU::precompute start\n");
    if (m_constraints.empty()) throw std::runtime_error("ADMMImplCPU::precompute Error: empty constraints");
    int m = static_cast<int>(m_constraints.size());
    int nDynVert = static_vert_begin;

    m_M = SpMatf(nDynVert, nDynVert);
    TripList M_trips;
    for (int vi = 0; vi < nDynVert; vi++) M_trips.emplace_back(vi, vi, m_M_vec(vi));
    m_M.setFromTriplets(M_trips.begin(), M_trips.end());

    m_dt2DTWeTWeD = SpMatf(nDynVert, nDynVert);
    m_dt2DTWeTWe = SpMatf(nDynVert, m_nCDim);
    m_D = SpMatf(m_nCDim, nDynVert);
    m_W_e = SpMatf(m_nCDim, m_nCDim);
    TripList D_trips, We_trips;
    for (int ci = 0; ci < m; ci++) {
        const auto& ct = m_constraints[ci];
        ct->get_D(D_trips, false);
        for (int i = 0; i < ct->dim; i++) We_trips.emplace_back(ct->start_row + i, ct->start_row + i, ct->w);
    }
    m_D.setFromTriplets(D_trips.begin(), D_trips.end());
    m_W_e.setFromTriplets(We_trips.begin(), We_trips.end());
    m_dt2DTWeTWe = m_dt2 * m_D.transpose() * m_W_e.transpose() * m_W_e;
    m_dt2DTWeTWeD = m_dt2DTWeTWe * m_D;

    m_Damp_Mat = SpMatf(nDynVert, nDynVert);
    TripList D_d_trips, We_d_trips;
    for (int ci = 0; ci < m; ci++) {
        const auto& ct = m_constraints[ci];
        if (ct->type >= 4) continue;
        ct->get_D(D_d_trips, false);
        for (int i = 0; i < ct->dim; i++) We_d_trips.emplace_back(ct->start_row + i, ct->start_row + i, ct->w);
    }
    SpMatf Wd(m_nCDim, m_nCDim), Dd(m_nCDim, nDynVert);
    Wd.setFromTriplets(We_d_trips.begin(), We_d_trips.end());
    Dd.setFromTriplets(D_d_trips.begin(), D_d_trips.end());
    m_Damp_Mat = m_dt * (damp_k_L * Dd.transpose() * Wd * Dd + damp_k_M * m_M);

    m_dt2Wc = SpMatf(nDynVert, nDynVert);
    m_W_c = SpMatf(nDynVert, nDynVert);
    if (enable_frictional_contact) {
        TripList Wc_trips;
        if (use_heuristic_W_c) {
            printf("use heuriostic W_c solution:\n");
            for (int i = 0; i < nDynVert; i++) {
                if (vinds_surf_set.count(i)) {
                    Real eigen_val_ii = m_A_damp.coeff(i, i);
                    Real m_ii = m_M_vec(i);
                    Real w_c_i = std::max(heu_sigma * eigen_val_ii, std::min(heu_beta * m_ii, eigen_val_ii));
                    w_c_i *= m_dt_inv * m_dt_inv * 10;
                    contact_w_list[i] = w_c_i;
                    contact_w_inv_list[i] = 1.0_r / w_c_i;
                }
            }
        }
        for (int i = 0; i < nDynVert; i++) {
            if (vinds_surf_set.count(i)) Wc_trips.emplace_back(i, i, contact_w_list(i));
        }
        m_W_c.setFromTriplets(Wc_trips.begin(), Wc_trips.end());
        m_dt2Wc = m_dt2 * m_W_c;
    } else {
        m_dt2Wc.setZero();
    }

    m_A = SpMatf(nDynVert, nDynVert);
    m_A = m_M + m_dt2DTWeTWeD + m_dt2Wc;
    m_A_damp = m_A + m_Damp_Mat;
    m_I = SpMatf(nDynVert, nDynVert);
    m_I.setIdentity();

    m_Ue = Matf_X3(m_nCDim, 3);
    m_Ue.setZero();
    m_Uc = Matf_X3(nDynVert, 3);
    m_Uc.setZero();
    if (enable_frictional_contact && prox_query) Gamma_c.reserve(prox_query->max_collision_num);

    m_LLT_solver = std::make_unique<SolverT>();
    m_LLT_solver->compute(m_A_damp);
    if (m_LLT_solver->info() != Eigen::Success) make_exception("ADMMImplCPU::precompute LLT failed");

    auto save_name_A = "./m_A_damp_" + getCurTime();
    if (Eigen::saveMarket(m_A_damp, save_name_A)) printf("saved m_A_damp\n");
    else printf("save m_A_damp failed\n");

    printf("ADMMImplCPU done, total vert num = %d\n", nDynVert);
}

void ADMMImplCPU::step() {
    step_fast();
}

void ADMMImplCPU::step_fast() {
    using namespace ADU;
    epsilon = 1.0_r / (m_dt2 * kappa + m_dt * beta);
    gamma = m_dt * kappa / (m_dt * kappa + beta);
    int nDynVert = static_vert_begin;

    Matf_X3 x_0 = m_vertices;
    Matf_X3 v_0 = m_velocities;
    for (int i = 0; i < nDynVert; i++) {
        if (m_pin_inds_set.count(i)) v_0.row(i).setZero();
        else v_0.row(i).y() -= g * m_dt;
    }

    Matf_X3 x_tilde = x_0.block(0, 0, nDynVert, 3) + m_dt * v_0.block(0, 0, nDynVert, 3);
    Matf_X3 M_x_tilde = m_M * x_tilde;
    Matf_X3 b(nDynVert, 3);
    b.setZero();

    Matf_X3 x_curr(m_nVert, 3);
    x_curr.block(0, 0, nDynVert, 3) = x_tilde;
    x_curr.block(nDynVert, 0, m_nVert - nDynVert, 3) = x_0.block(nDynVert, 0, m_nVert - nDynVert, 3);
    Matf_X3 z(m_nCDim, 3);
    z.setZero();
    Matf_X3 DX(m_nCDim, 3);
    DX.setZero();

    if (animator) animator->animate_all(m_vertices, x_curr, v_0, this->m_dt);
    Matf_X3 b_ini = m_Damp_Mat * x_0.block(0, 0, nDynVert, 3);
    Matf_X3 p(m_nVert, 3);
    p = v_0;

    if (!warmstart_Ue) m_Ue.setZero();
    if (!warmstart_Uc) m_Uc.setZero();

    Timer timer("ADMMImplCPU::step()");
    x_prev = x_curr;

    for (int admm_it = 0; admm_it < admm_max_iter; admm_it++) {
        if (enable_frictional_contact && prox_query && (admm_it % collision_detection_interval == 0)) {
            if (use_CCD) {
#ifdef USE_CCD
                prox_query->proximal_query_with_CCD(x_0, x_curr);
#else
                prox_query->proximal_query(x_curr);
#endif
            } else {
                prox_query->proximal_query(x_curr);
            }
            if (use_unique_contact) prox_query->unique_contact();
            need_recompute_Scc = true;
            find_contact_islands();
        }

        x_prev = x_curr;
        tbb::parallel_invoke(
            [&]() {
                DX = m_D * x_curr.block(0, 0, nDynVert, 3);
                z = DX + m_Ue;
                tbb::parallel_for_each(m_constraints.begin(), m_constraints.end(), [&](const std::shared_ptr<Constraint>& ct) {
                    int cdim = ct->dim;
                    Matf_XX zi = z.block(ct->start_row, 0, cdim, 3);
                    ct->prox(zi);
                    z.block(ct->start_row, 0, cdim, 3) = zi;
                });
            },
            [&]() {
                if (enable_frictional_contact) {
                    p.block(0, 0, nDynVert, 3) =
                        (x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) + m_Uc) * m_dt_inv;
                    if (need_recompute_Scc) {
                        compute_Scc();
                        need_recompute_Scc = false;
                    }
                    project_feasible(p, prox_query->contact_info_list, this->mu, gs_max_iter);
                }
            });

        b = M_x_tilde + m_dt2DTWeTWe * (z - m_Ue);
        if (enable_frictional_contact)
            b += m_dt2Wc * (m_dt * p.block(0, 0, nDynVert, 3) + x_0.block(0, 0, nDynVert, 3) - m_Uc);
        b += b_ini;

        tbb::parallel_invoke(
            [&]() { x_curr.block(0, 0, nDynVert, 1) = m_LLT_solver->solve(b.col(0)); },
            [&]() { x_curr.block(0, 1, nDynVert, 1) = m_LLT_solver->solve(b.col(1)); },
            [&]() { x_curr.block(0, 2, nDynVert, 1) = m_LLT_solver->solve(b.col(2)); });

        DX = m_D * x_curr.block(0, 0, nDynVert, 3);
        m_Ue += DX - z;
        m_Uc += x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) - p.block(0, 0, nDynVert, 3) * m_dt;
    }

    m_velocities.block(0, 0, nDynVert, 3) = (x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3)) / m_dt;
    m_velocities.block(nDynVert, 0, m_nVert - nDynVert, 3) = v_0.block(nDynVert, 0, m_nVert - nDynVert, 3);
    m_vertices = x_curr;
    step_cnt++;
}

void ADMMImplCPU::compute_Scc() {
    using namespace ADU;
    auto& contacts = prox_query->contact_info_list;
    size_t nContact = contacts.size();
    Gamma_c.resize(nContact, Vecf_4::Zero());
    K_c.resize(nContact, Vecf_3::Zero());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t ci = r.begin(); ci < r.end(); ci++) {
            const auto& ct = contacts[ci];
            Real Sc = epsilon;
            for (int i = 0; i < 4; i++) Sc += ct.bary[i] * ct.bary[i] * (contact_w_inv_list(ct.vinds[i]));
            for (int i = 0; i < 4; i++) {
                if (contact_w_list(ct.vinds[i]) > 1e7_r) Gamma_c[ci](i) = 0;
                else Gamma_c[ci](i) = ct.bary[i] / (Sc * contact_w_list(ct.vinds[i]));
            }
            K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma;
        }
    });
}

void ADMMImplCPU::project_feasible(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_GS_iter) {
    _project_feasible_plain(p, contacts, mu, max_GS_iter);
}

void ADMMImplCPU::_project_feasible_plain(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_GS_iter) {
    using namespace ADU;
    size_t nContact = contacts.size();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t ci = r.begin(); ci < r.end(); ci++) {
            const auto& ct = contacts[ci];
            for (size_t k = 0; k < 4; k++) {
                size_t j = ct.vinds[k];
                if (j < static_cast<size_t>(static_vert_begin)) p.row(j) += Gamma_c[ci](k) * ct.r_c;
            }
        }
    });
    Vecf_3 u, u_star, u_star_new, u_star_T;
    Real u_star_N, tau, alpha;
    for (size_t gs_iter = 0; gs_iter < max_GS_iter; gs_iter++) {
        for (size_t ci = 0; ci < nContact; ci++) {
            auto& ct = contacts[ci];
            const auto& vinds = ct.vinds;
            u = ct.bary[0] * p.row(vinds[0]) + ct.bary[1] * p.row(vinds[1]) +
                ct.bary[2] * p.row(vinds[2]) + ct.bary[3] * p.row(vinds[3]);
            u += K_c[ci];
            u_star = u - ct.r_c;
            const auto& normal = ct.normal;
            u_star_N = normal.dot(u_star);
            u_star_T = u_star - u_star_N * normal;
            if (u_star_N < 0) {
                if (!use_anisotropy) {
                    tau = u_star_T.norm();
                    alpha = -mu * u_star_N;
                    u_star_N = 0;
                    if (tau <= alpha) u_star_T = Vecf_3::Zero();
                    else u_star_T = (1.0_r - alpha / tau) * u_star_T;
                } else {
                    Real mu_t_curr = ct.d * mu_t + (1 - ct.d) * mu;
                    Real mu_b_curr = ct.d * mu_b + (1 - ct.d) * mu;
                    const Vecf_3& t = (ct.Sm - ct.Sm.dot(normal) * normal).normalized();
                    const Vecf_3& b = normal.cross(t).normalized();
                    Real tau_t = u_star_T.dot(t), tau_b = u_star_T.dot(b);
                    Real alpha_t = -mu_t_curr * u_star_N, alpha_b = -mu_b_curr * u_star_N;
                    u_star_N = 0;
                    u_star_T.setZero();
                    if (std::abs(tau_b) > alpha_b) u_star_T += (1.0_r - alpha_b / std::abs(tau_b)) * tau_b * b;
                    if (std::abs(tau_t) > alpha_t) u_star_T += (1.0_r - alpha_t / std::abs(tau_t)) * tau_t * t;
                }
            }
            u_star_new = u_star_N * ct.normal + u_star_T;
            ct.r_c += u_star_new - u;
            for (size_t k = 0; k < 4; k++) {
                size_t j = ct.vinds[k];
                if (j < static_cast<size_t>(static_vert_begin)) p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
            }
        }
    }
}

void ADMMImplCPU::find_contact_islands() {}

void ADMMImplCPU::compute_SaSb() {}

} // namespace ADU
