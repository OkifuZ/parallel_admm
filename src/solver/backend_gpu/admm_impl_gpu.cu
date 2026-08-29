/// Phase 3.4: GPU ADMM implementation (migrated from admm_parallel_global_solver.cu).

#include "solver/backend_gpu/admm_impl_gpu.h"
#include "solver/backend_gpu/linear_solver_gpu.h"
#include "solver/backend_gpu/local_projector_gpu.h"
#include "mutils/timer.h"
#include "constraint/pin_constraint.h"
#include "constraint/bending_constraint.h"
#include "thrust/device_vector.h"
#include "solver/contact_solver.cuh"
#include "solver/jacobi_solver.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#include <memory>

ContactDataDevice::ContactDataDevice(int max_Vert, int max_Contact) {
    d_vi_ct_count.reserve(max_Vert + 10);
    d_vi_ct_nums.reserve(max_Vert + 10);
    d_start_idx.reserve(max_Vert + 10);
    d_contact_W_list.reserve(max_Vert + 10);
    d_contact_W_inv_list.reserve(max_Vert + 10);
    d_bary.reserve(max_Contact);
    d_normal.reserve(max_Contact);
    d_point.reserve(max_Contact);
    d_h_cN.reserve(max_Contact);
    d_pair_type.reserve(max_Contact);
    d_Gamma_c.reserve(max_Contact);
    d_K_c.reserve(max_Contact);
    d_delta_u.reserve(max_Contact);
    d_r_c.reserve(max_Contact);
    d_involved_cid.reserve(max_Vert * 30);
    d_Gamma_i.reserve(max_Vert * 30);
}

namespace ADU {

ADMMImplGPU::ADMMImplGPU() {
    // Replace the CPU sub-solvers with the device versions.
    linear_solver_ = std::make_unique<LinearSolverGPU>(this);
    local_projector_ = std::make_unique<LocalProjectorGPU>(this);
}

ADMMImplGPU::~ADMMImplGPU() = default;

/// Device elastic prox: z = D*x + Ue, then per-type run_proxy kernels.
void ADMMImplGPU::do_project_elastic() {
    op_Ax(*m_D_device, solver_data_device->x_curr_device, DX_device);
    op_a_plus_b(DX_device, m_Ue_device, solver_data_device->z_buffer, m_nCDim);
    tet_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + tet_constraint_start_row * 3));
    triangle_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + triangle_constraint_start_row * 3));
    bending_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + bending_constraint_start_row * 3));
    pin_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + pin_constraint_start_row * 3));
}

void ADMMImplGPU::convert_constraint2device() {
    std::vector<std::shared_ptr<TriangleConstraint>> tri_constraints;
    std::vector<std::shared_ptr<PinConstraint>> pin_constraints;
    std::vector<std::shared_ptr<BendingConstraint>> bend_constraints;
    std::vector<std::shared_ptr<TetrahedralConstraint>> tet_constraints;

    for (auto& ct : m_constraints) {
        if (ct->type == 1) {
            static bool hit1 = false;
            if (!hit1) { hit1 = true; triangle_constraint_start_row = ct->start_row; }
            tri_constraints.push_back(std::dynamic_pointer_cast<TriangleConstraint>(ct));
        } else if (ct->type == 2) {
            static bool hit2 = false;
            if (!hit2) { hit2 = true; tet_constraint_start_row = ct->start_row; }
            tet_constraints.push_back(std::dynamic_pointer_cast<TetrahedralConstraint>(ct));
        } else if (ct->type == 3) {
            static bool hit3 = false;
            if (!hit3) { hit3 = true; bending_constraint_start_row = ct->start_row; }
            bend_constraints.push_back(std::dynamic_pointer_cast<BendingConstraint>(ct));
        } else if (ct->type == 4) {
            static bool hit4 = false;
            if (!hit4) { hit4 = true; pin_constraint_start_row = ct->start_row; }
            pin_constraints.push_back(std::dynamic_pointer_cast<PinConstraint>(ct));
        }
    }

    std::cout << "tri sr: " << triangle_constraint_start_row << std::endl;
    std::cout << "pin sr: " << pin_constraint_start_row << std::endl;
    std::cout << "bend sr: " << bending_constraint_start_row << std::endl;
    std::cout << "tet sr: " << tet_constraint_start_row << std::endl;

    pin_constraint_cu = std::make_unique<PinConstraintDevice>(pin_constraints);
    triangle_constraint_cu = std::make_unique<TriangleConstraintDevice>(tri_constraints);
    bending_constraint_cu = std::make_unique<BendingConstraintDevice>(bend_constraints);
    tet_constraint_cu = std::make_unique<TetrahedralConstraintDevice>(tet_constraints);
}

void ADMMImplGPU::init() {
    printf("ADMMImplGPU::init start\n");
    contact_normals.resize(prox_query->max_collision_num);
    contact_points.resize(prox_query->max_collision_num);
    ct_data_device = std::make_shared<ContactDataDevice>(m_nVert, prox_query->max_collision_num);
    precompute();

    int nDynVert = static_vert_begin;
    comp_mat = std::make_unique<CompactSparseMat>(m_A);
    solver_data_device = std::make_unique<CuSolverData>(*comp_mat, nDynVert, m_nVert, nDynVert, m_nVert, m_nCDim);

    jacobi_buffer.resize(nDynVert, 3);
    jacobi_buffer.setZero();

    if (enable_frictional_contact && prox_query) {
        Gamma_c.reserve(prox_query->max_collision_num);
        Gamma_i.resize(m_nVert);
        involved_cid.resize(m_nVert);
        for (int vi = 0; vi < m_nVert; vi++) {
            Gamma_i[vi].reserve(50);
            involved_cid[vi].reserve(50);
        }
        delta_u.reserve(prox_query->max_collision_num);
        vi_ct_nums.resize(m_nVert, 0);
    }

    resizeThrust<Real>(cache_nCDimX3, m_nCDim * 3, 0);
    resizeThrust<Real>(cache_nDynVertX3, nDynVert * 3, 0);
    resizeThrust<Real>(cache_nDynVertX3_bp1, nDynVert * 3, 0);

    std::vector<int> is_fixed(m_nVert, 0);
    for (auto x : m_pin_inds_set) is_fixed[x] = 1;
    resizeThrust(d_is_fixed, m_nVert, 0);
    thrust::copy(is_fixed.begin(), is_fixed.end(), d_is_fixed.begin());

    resizeThrust(d_M, m_nVert, 1000000.0f);
    copy_vec2thrustvector(m_M_vec, d_M, m_nVert);

    resizeThrust<Real>(M_x_tilde_device, m_nVert * 3, 0);
    resizeThrust<Real>(DX_device, m_nCDim * 3, 0);

    copy_mat2thrustvector(m_vertices, solver_data_device->x_0_device, m_nVert);
    copy_mat2thrustvector(m_velocities, solver_data_device->v_0_device, m_nVert);

    if (animator) {
        animator->reset();
        animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
        copy_mat2thrustvector(m_vertices, solver_data_device->x_0_device, nDynVert, m_nVert);
        copy_mat2thrustvector(m_velocities, solver_data_device->v_0_device, nDynVert, m_nVert);
    }
    printf("ADMMImplGPU init done\n");
}

void ADMMImplGPU::reset(const Matf_X3& ini_verts, bool need_precompute) {
    ADMMImplCPUBase::reset(ini_verts, need_precompute);
    copy_mat2thrustvector(this->m_vertices, solver_data_device->x_0_device, m_nVert);
    copy_mat2thrustvector(this->m_velocities, solver_data_device->v_0_device, m_nVert);
}

void ADMMImplGPU::precompute() {
    using namespace ADU;
    printf("ADMMImplGPU::precompute start\n");
    if (m_constraints.empty()) throw std::runtime_error("ADMMImplGPU::precompute Error: empty constraints");
    int m = static_cast<int>(m_constraints.size());
    int nDynVert = static_vert_begin;

    m_M = SpMatf(nDynVert, nDynVert);
    TripList M_trips;
    for (int vi = 0; vi < nDynVert; vi++) M_trips.emplace_back(vi, vi, m_M_vec(vi));
    m_M.setFromTriplets(M_trips.begin(), M_trips.end());
    m_M_device = std::make_unique<CompactSparseMat>(m_M, false);

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
    m_D_device = std::make_unique<CompactSparseMat>(m_D, false);

    m_dt2DTWeTWe = m_dt2 * m_D.transpose() * m_W_e.transpose() * m_W_e;
    m_dt2DTWeTWeD = m_dt2DTWeTWe * m_D;
    m_dt2DTWeTWe_device = std::make_unique<CompactSparseMat>(m_dt2DTWeTWe, false);

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
            SpMatf m_A_temp = m_M + m_dt2DTWeTWeD;
            SpMatf m_A_damp_temp = m_A_temp + m_Damp_Mat;
            for (int i = 0; i < nDynVert; i++) {
                if (vinds_surf_set.count(i)) {
                    Real eigen_val_ii = m_A_damp_temp.coeff(i, i);
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
    m_dt2Wc_device = std::make_unique<CompactSparseMat>(m_dt2Wc, false);

    if (enable_frictional_contact) {
        m_A = (m_M + m_dt2DTWeTWeD + m_dt2Wc).eval();
    } else {
        m_A = (m_M + m_dt2DTWeTWeD).eval();
    }

    m_I = SpMatf(nDynVert, nDynVert);
    m_I.setIdentity();
    m_Ue = Matf_X3(m_nCDim, 3);
    m_Ue.setZero();
    m_Uc = Matf_X3(nDynVert, 3);
    m_Uc.setZero();
    resizeThrust<Real>(m_Ue_device, 3 * m_nCDim, 0);
    resizeThrust<Real>(m_Uc_device, 3 * nDynVert, 0);

    if (enable_frictional_contact && prox_query) Gamma_c.reserve(prox_query->max_collision_num);

    printf("ADMMImplGPU done, total vert num = %d\n", nDynVert);

    resizeThrust<int>(ct_data_device->d_vi_ct_nums, m_nVert, 0);
    resizeThrust<Real>(ct_data_device->d_contact_W_list, m_nVert, 0);
    resizeThrust<Real>(ct_data_device->d_contact_W_inv_list, m_nVert, 0);
    copy_vec2thrustvector(contact_w_list, ct_data_device->d_contact_W_list, m_nVert);
    copy_vec2thrustvector(contact_w_inv_list, ct_data_device->d_contact_W_inv_list, m_nVert);
}

void ADMMImplGPU::step() {
    using namespace ADU;
    epsilon = 1.0_r / (m_dt2 * kappa + m_dt * beta);
    gamma = m_dt * kappa / (m_dt * kappa + beta);
    int nDynVert = static_vert_begin;

    x_curr.resize(m_nVert, 3);
    x_curr.setZero();

    if (animator) {
        animator->animate_all(m_vertices, x_curr, m_velocities, this->m_dt);
        copy_mat2thrustvector(x_curr, solver_data_device->x_0_device, nDynVert, m_nVert);
        copy_mat2thrustvector(m_velocities, solver_data_device->v_0_device, nDynVert, m_nVert);
    }

    do_pre_integration(m_nVert, nDynVert, m_dt, g, solver_data_device->x_0_device.data().get(),
        d_is_fixed.data().get(), d_M.data().get(), solver_data_device->v_0_device.data().get(),
        solver_data_device->x_curr_device.data().get(), M_x_tilde_device.data().get());

    if (!warmstart_Ue) m_Ue.setZero();
    if (!warmstart_Uc) m_Uc.setZero();

    Timer timer("ADMMImplGPU::step()");
    float dt_r = static_cast<float>(m_dt);
    float dt_r_inv = 1.0f / dt_r;
    copy_thrustvector2thrust(solver_data_device->v_0_device, solver_data_device->p_device, nDynVert * 3, m_nVert * 3);

    for (int admm_it = 0; admm_it < admm_max_iter; admm_it++) {
        if (enable_frictional_contact && prox_query && (admm_it % collision_detection_interval == 0)) {
            bvh->update(solver_data_device->x_curr_device.data().get(), m_nVert, admm_it + collision_detection_interval >= admm_max_iter);
            bvh->dcd();
            convert_DCD_info(bvh, solver_data_device->x_curr_device, ct_data_device.get());
            need_recompute_Scc = true;
        }

        {
            local_projector_->project_elastic();

            if (enable_frictional_contact) {
                op_a_minus_b(solver_data_device->x_curr_device, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
                op_a_plus_b(cache_nDynVertX3, m_Uc_device, cache_nDynVertX3, nDynVert);
                op_scale(cache_nDynVertX3, dt_r_inv, solver_data_device->p_device, nDynVert);
                if (need_recompute_Scc) {
                    compute_Scc_parallel();
                    need_recompute_Scc = false;
                }
                project_feasible_parallel();
            }
        }

        {
            op_a_minus_b(solver_data_device->z_buffer, m_Ue_device, cache_nCDimX3, m_nCDim);
            op_Ax(*m_dt2DTWeTWe_device, cache_nCDimX3, cache_nDynVertX3);
            op_a_plus_b(M_x_tilde_device, cache_nDynVertX3, solver_data_device->b_curr_device, nDynVert);

            if (enable_frictional_contact) {
                op_scale(solver_data_device->p_device, dt_r, cache_nDynVertX3, nDynVert);
                op_a_plus_b(cache_nDynVertX3, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
                op_a_minus_b(cache_nDynVertX3, m_Uc_device, cache_nDynVertX3, nDynVert);
                op_Ax(*m_dt2Wc_device, cache_nDynVertX3, cache_nDynVertX3);
                op_a_plus_b(solver_data_device->b_curr_device, cache_nDynVertX3, solver_data_device->b_curr_device, nDynVert);
            }
            linear_solver_->solve();
        }

        op_Ax(*m_D_device, solver_data_device->x_curr_device, DX_device);
        op_a_minus_b(DX_device, solver_data_device->z_buffer, cache_nCDimX3, m_nCDim);
        op_a_plus_b(cache_nCDimX3, m_Ue_device, m_Ue_device, m_nCDim);

        if (enable_frictional_contact) {
            op_a_minus_b(solver_data_device->x_curr_device, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
            op_scale(solver_data_device->p_device, dt_r, cache_nDynVertX3_bp1, nDynVert);
            op_a_minus_b(cache_nDynVertX3, cache_nDynVertX3_bp1, cache_nDynVertX3, nDynVert);
            op_a_plus_b(m_Uc_device, cache_nDynVertX3, m_Uc_device, nDynVert);
        }
    }

    do_post_process(m_nVert, nDynVert, m_dt, solver_data_device->v_0_device.data().get(),
        solver_data_device->x_curr_device.data().get(), solver_data_device->x_0_device.data().get());

    if (sync_to_host) {
        copy_thrustvector2mat(solver_data_device->x_0_device, m_vertices, m_nVert);
        copy_thrustvector2mat(solver_data_device->v_0_device, m_velocities, m_nVert);
    }
    step_cnt++;
}

void ADMMImplGPU::GS_global(const Matf_X3& b, Matf_X3& x_curr, int iter_cnt) {
    int nDynVert = static_vert_begin;
    for (int i = 0; i < iter_cnt; i++) {
        for (int i = 0; i < nDynVert; i++) {
            Real a_ii = comp_mat->diag_data[i];
            Vecf_3 b_i_p = b.row(i);
            Vecf_3 b_i_m = Vecf_3::Zero();
            int st = comp_mat->offdiag_perline_start[i];
            int ed = comp_mat->offdiag_perline_start[i + 1];
            for (int j = st; j < ed; j++) {
                int ind = comp_mat->offdiag_indices[j];
                b_i_m += x_curr.row(ind) * comp_mat->offdiag_data[j];
            }
            x_curr.row(i) = (b_i_p - b_i_m) / a_ii;
        }
    }
}

void ADMMImplGPU::Jacobi_global(int iter_cnt) {
    int nDynVert = static_vert_begin;
    cu_jacobi_global(solver_data_device->sp_mat_device, solver_data_device->b_curr_device,
        solver_data_device->x_curr_device, solver_data_device->jacobi_buffer_1,
        solver_data_device->jacobi_buffer_2, nDynVert, iter_cnt);
}

void ADMMImplGPU::project_feasible_parallel() {
    _project_feasible_impl();
}

void ADMMImplGPU::_project_feasible_impl() {
    project_impl(bvh, ct_data_device.get(), solver_data_device.get(), gs_max_iter, static_vert_begin, mu, static_vert_begin);
}

void ADMMImplGPU::_project_feasible_plain(Matf_X3& p, ProximalQuery::ContactInfoList& contacts, Real mu, size_t max_iter) {
    using namespace ADU;
    size_t nContact = contacts.size();
    for (size_t ci = 0; ci < nContact; ci++) {
        const auto& ct = contacts[ci];
        for (size_t k = 0; k < 4; k++) {
            size_t j = ct.vinds[k];
            if (j < static_cast<size_t>(static_vert_begin)) p.row(j) += Gamma_c[ci](k) * ct.r_c;
        }
    }
    for (size_t gs_iter = 0; gs_iter < max_iter; gs_iter++) {
        tbb::task_arena task_arena(16);
        task_arena.execute([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](tbb::blocked_range<size_t> r) {
                Vecf_3 u, u_star, u_star_new, u_star_T;
                Real u_star_N, tau, alpha;
                for (size_t ci = r.begin(); ci < r.end(); ci++) {
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
                        tau = u_star_T.norm();
                        alpha = -mu * u_star_N;
                        u_star_N = 0;
                        if (tau <= alpha) u_star_T = Vecf_3::Zero();
                    else u_star_T = (1.0_r - alpha / tau) * u_star_T;
                    }
                    u_star_new = u_star_N * ct.normal + u_star_T;
                    delta_u[ci] = u_star_new - u;
                    ct.r_c += delta_u[ci];
                }
            });
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_nVert), [&](tbb::blocked_range<size_t> r) {
                for (size_t vi = r.begin(); vi < r.end(); vi++) {
                    Vecf_3 delta_pi = Vecf_3::Zero();
                    const auto& vcids = involved_cid[vi];
                    for (size_t cidx = 0; cidx < vcids.size(); cidx++) {
                        int cid = vcids[cidx];
                        delta_pi += delta_u[cid] * Gamma_i[vi][cidx];
                    }
                    p.row(vi) += delta_pi;
                }
            });
        });
    }
}

void ADMMImplGPU::compute_Scc_parallel() {
    compute_Scc_impl();
}

void ADMMImplGPU::compute_Scc() {
    using namespace ADU;
    auto& contacts = prox_query->contact_info_list;
    size_t nContact = contacts.size();
    Gamma_c.resize(nContact, Vecf_4::Zero());
    K_c.resize(nContact, Vecf_3::Zero());
    for (int vi = 0; vi < m_nVert; vi++) {
        Gamma_i[vi].clear();
        involved_cid[vi].clear();
    }
    std::fill(vi_ct_nums.begin(), vi_ct_nums.end(), 0);
    delta_u.resize(nContact, Vecf_3::Zero());

    for (size_t ci = 0; ci < nContact; ci++) {
        const auto& c = contacts[ci];
        vi_ct_nums[c.vinds[0]]++;
        vi_ct_nums[c.vinds[1]]++;
        vi_ct_nums[c.vinds[2]]++;
        vi_ct_nums[c.vinds[3]]++;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t ci = r.begin(); ci < r.end(); ci++) {
            const auto& ct = contacts[ci];
            Real Sc = epsilon;
            for (int i = 0; i < 4; i++) {
                size_t vi = ct.vinds[i];
                Sc += ct.bary[i] * ct.bary[i] * contact_w_inv_list(vi) * vi_ct_nums[vi];
            }
            for (int i = 0; i < 4; i++) {
                int vi = ct.vinds[i];
                if (contact_w_list(ct.vinds[i]) > 1e7_r) {
                    Gamma_c[ci](i) = 0;
                    Gamma_i[vi].emplace_back(0);
                } else {
                    Gamma_c[ci](i) = ct.bary[i] / (Sc * contact_w_list(vi));
                    Gamma_i[vi].emplace_back(ct.bary[i] / (Sc * contact_w_list(vi)));
                }
                involved_cid[vi].push_back(static_cast<int>(ci));
            }
            K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma;
        }
    });
}

void ADMMImplGPU::compute_Scc_impl() {
    size_t nContact = bvh->h_cpNum;
    computeContactCountsWithAtomic(bvh->d_collisonPairs, ct_data_device->d_vi_ct_nums, nContact, m_nVert);
    compute_Scc_impl_cu(bvh, nContact, this->m_dt_inv, gamma, epsilon, ct_data_device.get());
}

std::vector<std::array<float, 3>>& ADMMImplGPU::getContactPoints() {
    std::fill(contact_points.begin(), contact_points.end(), std::array<float, 3>{0, 0, 0});
    CUDA_SAFE_CALL(cudaMemcpy(contact_points.data(), ct_data_device->d_point.data().get(), sizeof(float) * 3 * bvh->h_cpNum, cudaMemcpyDeviceToHost));
    return contact_points;
}

std::vector<std::array<float, 3>>& ADMMImplGPU::getContactNormals() {
    std::fill(contact_normals.begin(), contact_normals.end(), std::array<float, 3>{0, 0, 0});
    CUDA_SAFE_CALL(cudaMemcpy(contact_normals.data(), ct_data_device->d_normal.data().get(), sizeof(float) * 3 * bvh->h_cpNum, cudaMemcpyDeviceToHost));
    return contact_normals;
}

} // namespace ADU
