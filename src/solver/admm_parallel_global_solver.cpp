#include "solver/admm_full_solver.h"
#include "mutils/timer.h"
#include "constraint/pin_constraint.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <mutex>
#include <memory>
#include <random>

void ADMMParallelSolver::init() {
    ADMMSolverFull_RL_damping::init();
    comp_mat = std::make_unique<CompactSparseMat>(m_A_damp);
    
    int nDynVert = static_vert_begin;
    jacobi_buffer.resize(nDynVert, 3);

	if (enable_frictional_contact && prox_query) {
		Gamma_c.reserve(prox_query->max_collision_num);
		Gamma_i.resize(m_nVert);
		involved_cid.resize(m_nVert);

		const int reserved_cts = 50;

		for (int vi = 0; vi < m_nVert; vi++) {
			Gamma_i[vi].reserve(reserved_cts);
			involved_cid[vi].reserve(reserved_cts);
		}
		delta_u.reserve(prox_query->max_collision_num);
		vi_ct_nums.resize(m_nVert, 0);
	}

	
}

void ADMMParallelSolver::step() {
    using namespace ADU;
    printf("here");

    // ADMMSolverFull_RL_damping::step();

	// for contact stabilization
	epsilon = 1.0_r / (m_dt2 * kappa + m_dt * beta);
	gamma = m_dt * kappa / (m_dt * kappa + beta);

	int nDynVert = static_vert_begin;

	Matf_X3 x_0(m_nVert, 3);
	Matf_X3 v_0(m_nVert, 3);
	x_0 = m_vertices;
	v_0 = m_velocities;

	// explicit integrate as initial value
	for (int i = 0; i < nDynVert; i++) {
		if (m_pin_inds_set.count(i)) v_0.row(i).setZero();
		else v_0.row(i).y() -= g * m_dt;
	}

	Matf_X3 x_tilde = x_0.block(0, 0, nDynVert, 3) + m_dt * v_0.block(0, 0, nDynVert, 3); // nDynVert * 3
	Matf_X3 M_x_tilde = m_M * x_tilde; // nDynVert * 3
	Matf_X3 b = Matf_X3(nDynVert, 3);
	b.setZero();

	// initial value
	Matf_X3 x_curr(m_nVert, 3);
	x_curr.block(0, 0, nDynVert, 3) = x_tilde;
	x_curr.block(nDynVert, 0, m_nVert - nDynVert, 3) = x_0.block(nDynVert, 0, m_nVert - nDynVert, 3);
	Matf_X3 z(m_nCDim, 3);
	z.setZero();
	Matf_X3 DX(m_nCDim, 3);
	DX.setZero();

	if (animator) {
		animator->animate_all(m_vertices, x_curr, v_0, this->m_dt);
	}
	Matf_X3 b_ini = m_Damp_Mat * x_0.block(0, 0, nDynVert, 3);

	Matf_X3 p(m_nVert, 3); // this should be n*3
	p = v_0;

	if (!warmstart_Ue) { m_Ue.setZero(); }
	if (!warmstart_Uc) { m_Uc.setZero(); }

	Timer timer("ADMMSolverFull::step()");

	for (int admm_it = 0; admm_it < admm_max_iter; admm_it++) {
		Timer per_iteration_timer("per_iteration");

		// collision
		if (enable_frictional_contact && prox_query && (admm_it % collision_detection_interval == 0)) {
			Timer collision_timer("dynamic_collision_detection");
			//prox_query->proximal_query(x_curr);
			//if (use_CCD) {
			//	#ifdef USE_CCD
			//	prox_query->proximal_query_with_CCD(x_0, x_curr);
			//	//prox_query->proximal_query_with_CCD(x_tilde, x_curr);
			//	#else 
			//	prox_query->proximal_query(x_curr);
			//	#endif
			//}
			//else {
			//	prox_query->proximal_query(x_curr);
			//}

			//if (use_unique_contact) {
			//	prox_query->unique_contact();
			//}

			need_recompute_Scc = true;
			find_contact_islands(); // TODO

			bvh->update(x_curr, true);
			bvh->dcd();
			bvh->convert_contactInfo_device2host(prox_query->contact_info_list, x_curr);
		}

		{
			Timer local_project_timer("elastic_local & contact_local");
			tbb::parallel_invoke(
				[&]() {
					//Timer local_project_timer("elastic_local");
					// local
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
					if (enable_frictional_contact)
					{ // frictional contact
						//Timer local_project_timer("contact_local");

						p.block(0, 0, nDynVert, 3) =
							(x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) + m_Uc) * m_dt_inv; // p as start velocity 

						if (need_recompute_Scc) {
							compute_Scc();
							need_recompute_Scc = false;
						}

						// project p into feasible set
						project_feasible(p, prox_query->contact_info_list, this->mu, gs_max_iter);
					}
				}
			);
		}

		{
			Timer local_project_timer("global_propogation");

			// global
			b = M_x_tilde + m_dt2DTWeTWe * (z - m_Ue);
			// friction
			if (enable_frictional_contact) {
				b += m_dt2Wc * (m_dt * p.block(0, 0, nDynVert, 3) + x_0.block(0, 0, nDynVert, 3) - m_Uc);
			}
			// damp
			b += b_ini;

			for (int i = 0; i < 25; i++) {
                // GS_global(b, x_curr);
                Jacobi_global(b, x_curr);
            }
            // tbb::parallel_invoke(
			// 	[&]() {
			// 		x_curr.block(0, 0, nDynVert, 1) = m_LLT_solver->solve(b.col(0));
			// 	},
			// 	[&]() {
			// 		x_curr.block(0, 1, nDynVert, 1) = m_LLT_solver->solve(b.col(1));
			// 	},
			// 	[&]() {
			// 		x_curr.block(0, 2, nDynVert, 1) = m_LLT_solver->solve(b.col(2));
			// 	}
			// );	
		}
		DX = m_D * x_curr.block(0, 0, nDynVert, 3);
		m_Ue += DX - z;
		m_Uc += x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) - p.block(0, 0, nDynVert, 3) * m_dt;
	}
	
	m_velocities.block(0, 0, nDynVert, 3) = (x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3)) / m_dt;
	m_velocities.block(nDynVert, 0, m_nVert - nDynVert, 3) = v_0.block(nDynVert, 0, m_nVert - nDynVert, 3);
	m_vertices = x_curr;

	step_cnt++;

	
}

void ADMMParallelSolver::GS_global(const ADU::Matf_X3& b, ADU::Matf_X3& x_curr) {
    using namespace ADU;

    int nDynVert = static_vert_begin;
    for (int i = 0; i < nDynVert; i++) {
        Real a_ii = comp_mat->diag_data[i];
        Vecf_3 b_i_p = b.row(i); 
        
        Vecf_3 b_i_m = Vecf_3::Zero();
        int st_ind = comp_mat->offdiag_perline_start[i];
        int ed_ind = comp_mat->offdiag_perline_start[i+1];
        for (int j = st_ind; j < ed_ind; j++) {
            int ind = comp_mat->offdiag_indices[j];
            b_i_m += x_curr.row(ind) * comp_mat->offdiag_data[j];
        }

        x_curr.row(i) = (b_i_p - b_i_m) / a_ii;
    }

}

void ADMMParallelSolver::Jacobi_global(const ADU::Matf_X3& b, ADU::Matf_X3& x_curr) {
    using namespace ADU;

    int nDynVert = static_vert_begin;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, nDynVert), 
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                Real a_ii = comp_mat->diag_data[i];
                Vecf_3 b_i_p = b.row(i); 
                
                Vecf_3 b_i_m = Vecf_3::Zero();
                int st_ind = comp_mat->offdiag_perline_start[i];
                int ed_ind = comp_mat->offdiag_perline_start[i+1];
                for (int j = st_ind; j < ed_ind; j++) {
                    int ind = comp_mat->offdiag_indices[j];
                    b_i_m += x_curr.row(ind) * comp_mat->offdiag_data[j];
                }

                jacobi_buffer.row(i) = (b_i_p - b_i_m) / a_ii;
            }
    });
    x_curr.block(0, 0, nDynVert, 3) = jacobi_buffer;
    
    // for (int i = 0; i < nDynVert; i++) {
    //     Real a_ii = comp_mat->diag_data[i];
    //     Vecf_3 b_i_p = b.row(i); 
        
    //     Vecf_3 b_i_m = Vecf_3::Zero();
    //     int st_ind = comp_mat->offdiag_perline_start[i];
    //     int ed_ind = comp_mat->offdiag_perline_start[i+1];
    //     for (int j = st_ind; j < ed_ind; j++) {
    //         int ind = comp_mat->offdiag_indices[j];
    //         b_i_m += x_curr.row(ind) * comp_mat->offdiag_data[j];
    //     }

    //     jacobi_buffer.row(i) = (b_i_p - b_i_m) / a_ii;
    // }

    // x_curr.block(0, 0, nDynVert, 3) = jacobi_buffer;

}

void ADMMParallelSolver::_project_feasible_plain(ADU::Matf_X3& p,
		ProximalQuery::ContactInfoList& contacts,
		ADU::Real mu, size_t max_jacobi_iter) 
{
	// p: current velocity
	// notion: r_c lies in contact_info
	using namespace ADU;

	// ADMMSolverFull_RL_damping::_project_feasible_plain(p, contacts, mu, max_jacobi_iter);
	// return;

	size_t nContact = contacts.size();

	for (int ci = 0; ci < nContact; ci++) {
		const auto& ct = contacts[ci];
		for (size_t k = 0; k < 4; k++) {
			size_t j = ct.vinds[k];
			p.row(j) += Gamma_c[ci](k) * ct.r_c;
		}
	}

	for (size_t gs_iter = 0; gs_iter < max_jacobi_iter; gs_iter++) {
		// stage 1: compute delta_u
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](tbb::blocked_range<size_t> r) {
			Vecf_3 u{};
			Vecf_3 u_star{};
			Vecf_3 u_star_new{};
			Real u_star_N{};
			Vecf_3 u_star_T{};
			Real tau{};
			Real alpha{};
			for (size_t ci = r.begin(); ci < r.end(); ci++) {
				auto& ct = contacts[ci];
				const auto& vinds = ct.vinds;
				u = ct.bary[0] * p.row(vinds[0]) +
					ct.bary[1] * p.row(vinds[1]) +
					ct.bary[2] * p.row(vinds[2]) +
					ct.bary[3] * p.row(vinds[3]);

				u += K_c[ci];

				u_star = u - ct.r_c;

				// project u_star to satisfy coulomb law
				const auto& normal = ct.normal;
				u_star_N = normal.dot(u);
				// u_star_N = normal.dot(u_star);
				u_star_T = u_star - u_star_N * normal;
				if (u_star_N < 0) {
					tau = u_star_T.norm();
					alpha = -mu * u_star_N;
					u_star_N = 0;
					if (tau <= alpha) {
						u_star_T = Vecf_3::Zero();
					}
					else {
						u_star_T = (1.0_r - alpha / tau) * u_star_T;
					}
				}
				// update force
				u_star_new = u_star_N * ct.normal + u_star_T;
				delta_u[ci] = u_star_new - u;
				ct.r_c += delta_u[ci];
			}
			});

		// stage 2: update(report) p using delta_u in parallel
		tbb::parallel_for(tbb::blocked_range<size_t>(0, m_nVert), [&](tbb::blocked_range<size_t> r) {
			ADU::Vecf_3 delta_pi{};
			for (size_t vi = r.begin(); vi < r.end(); vi++) {
				delta_pi = ADU::Vecf_3::Zero();
				const auto& vcids = involved_cid[vi];
				for (int cidx = 0; cidx < vcids.size(); cidx++) {
					int cid = vcids[cidx];
					delta_pi += delta_u[cid] * Gamma_i[vi][cidx];
				}
				// if (delta_pi.x() != 0 || delta_pi.y() != 0 || delta_pi.z() != 0)
				// 	printf("%f %f %f\n", delta_pi.x(), delta_pi.y(), delta_pi.z());

				p.row(vi) += delta_pi;
			}
			});
	}
}

void ADMMParallelSolver::compute_Scc(bool is_XPBD ) {
	using namespace ADU;
	auto& contacts = prox_query->contact_info_list;
	size_t nContact = contacts.size();

	Gamma_c.resize(nContact, Vecf_4::Zero());
	K_c.resize(nContact, Vecf_3::Zero());
	//Real Sc_W{};
	for (int vi = 0; vi < m_nVert; vi++) {
		Gamma_i[vi].clear();
		involved_cid[vi].clear();
	}
	std::fill(vi_ct_nums.begin(), vi_ct_nums.end(), 0);
	delta_u.resize(nContact, Vecf_3::Zero());

	for (int ci = 0; ci < nContact; ci++) {
		const auto& c = contacts[ci];
		vi_ct_nums[c.vinds[0]]++;
		vi_ct_nums[c.vinds[1]]++;
		vi_ct_nums[c.vinds[2]]++;
		vi_ct_nums[c.vinds[3]]++;
	}

	tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](const tbb::blocked_range<size_t>& r) {
		Real Sc{};
		for (size_t ci = r.begin(); ci < r.end(); ci++) {
			const auto& ct = contacts[ci];
			// TODO: only correct when w_j is all the same
			/*Sc_W = ct.bary.squaredNorm() + epsilon;
			Gamma_c[ci] = ct.bary / Sc_W;*/
			Sc = epsilon;
			//Sc = 0;
			for (int i = 0; i < 4; i++) {
				Sc += ct.bary[i] * ct.bary[i] * (contact_w_inv_list(ct.vinds[i]));
			}
			for (int i = 0; i < 4; i++) {
				int vi = ct.vinds[i];
				if (contact_w_list(ct.vinds[i]) > 1e7_r) {
					Gamma_c[ci](i) = 0;
					Gamma_i[vi].emplace_back(0);
				}
				else {
					Gamma_c[ci](i) = ct.bary[i] / (Sc * contact_w_list(ct.vinds[i]));
					Gamma_i[vi].emplace_back(ct.bary[i] / (Sc * contact_w_list(vi)));
				}
			}
			
			// precompute k
			// k_c = 1 / dt * h_cN * n
			K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma;
		}
		});

}