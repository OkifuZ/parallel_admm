#include "solver/admm_full_solver.h"
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


void ADMMSolverFull_RL_damping::init() {
	using namespace ADU;
	Timer timer("ADMMSolverFull_RL_damping::init()");

	this->precompute();

	if (animator) {
		animator->reset();
		animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
	}
	// GN
	int nDynVert = static_vert_begin;

	// profile
#ifdef PROFILE_RR
	x_prev.resize(nDynVert, 3);
#endif // PROFILE_RR

	printf("ADMMSolverFull_RL_damping init done\n");
}

void ADMMSolverFull_RL_damping::precompute() {
	using namespace ADU;

	printf("ADMMSolverFull_RL_damping::precompute start\n");


	// construct global matrix
	if (m_constraints.size() == 0) {
		throw std::runtime_error("ADMMSolverFull_RL_damping::precompute Error: empty constraints");
	}
	int m = m_constraints.size();

	int nDynVert = static_vert_begin;

	// Sparse Mass Matrix triplets
	m_M = SpMatf(nDynVert, nDynVert);
	TripList M_trips;
	for (int vi = 0; vi < nDynVert; vi++) {
		M_trips.emplace_back(vi, vi, m_M_vec(vi));
	}
	m_M.setFromTriplets(M_trips.begin(), M_trips.end());

	// Sparse Elastic Matrices triplets
	m_dt2DTWeTWeD = SpMatf(nDynVert, nDynVert);
	m_dt2DTWeTWe = SpMatf(nDynVert, m_nCDim);
	m_D = SpMatf(m_nCDim, nDynVert);
	m_W_e = SpMatf(m_nCDim, m_nCDim);

	TripList D_trips;
	TripList We_trips;
	for (int ci = 0; ci < m; ci++) {
		const auto& ct = m_constraints[ci];
		ct->get_D(D_trips, false);
 		for (int i = 0; i < ct->dim; i++) {
			We_trips.emplace_back(ct->start_row + i, ct->start_row + i, ct->w);
		}
	}
	m_D.setFromTriplets(D_trips.begin(), D_trips.end());
	m_W_e.setFromTriplets(We_trips.begin(), We_trips.end());

	m_dt2DTWeTWe = m_dt2 * m_D.transpose() * m_W_e.transpose() * m_W_e;
	m_dt2DTWeTWeD = m_dt2DTWeTWe * m_D;

	// damping
	m_Damp_Mat = SpMatf(nDynVert, nDynVert); // h * k * D_r
	TripList D_d_trips;
	TripList We_d_trips;
	for (int ci = 0; ci < m; ci++) {
		const auto& ct = m_constraints[ci];
		if (ct->type >= 4) continue;
		ct->get_D(D_d_trips, false);
		for (int i = 0; i < ct->dim; i++) {
			We_d_trips.emplace_back(ct->start_row + i, ct->start_row + i, ct->w);
		}
	}

	SpMatf Wd(m_nCDim, m_nCDim);
	SpMatf Dd(m_nCDim, nDynVert);
	Wd.setFromTriplets(We_d_trips.begin(), We_d_trips.end());
	Dd.setFromTriplets(D_d_trips.begin(), D_d_trips.end());
	m_Damp_Mat = m_dt * (damp_k_L * Dd.transpose() * Wd * Dd + damp_k_M * m_M);



	// frictional contact
	m_dt2Wc = SpMatf(nDynVert, nDynVert);
	m_W_c = SpMatf(nDynVert, nDynVert);


	if (enable_frictional_contact) {
		TripList Wc_trips;
		if (use_heuristic_W_c) {
			printf("use heuriostic W_c solution:\n");
			Real eigen_val_ii{};
			Real m_ii{};
			Real w_c_i{};
			for (int i = 0; i < nDynVert; i++) {
				if (vinds_surf_set.count(i)) {
					eigen_val_ii = m_A_damp.coeff(i, i);

					m_ii = m_M_vec(i);
					w_c_i = std::max(
						heu_sigma * eigen_val_ii, 
						std::min(heu_beta * m_ii, eigen_val_ii));
					w_c_i *= m_dt_inv * m_dt_inv * 10;
					contact_w_list[i] = w_c_i;
					contact_w_inv_list[i] = 1.0_r / w_c_i;
				}
			}
		}
		else {
			printf("not using heuriostic W_c solution:\n");
		}

		for (int i = 0; i < nDynVert; i++) {
			//Wc_trips.emplace_back(i, i, contact_w);
			if (vinds_surf_set.count(i)) {
				Real m_ii = m_M_vec(i);
				Wc_trips.emplace_back(i, i, contact_w_list(i));
			}
		}
		m_W_c.setFromTriplets(Wc_trips.begin(), Wc_trips.end());
		m_dt2Wc = m_dt2 * m_W_c;
	}
	else {
		m_dt2Wc.setZero();
	}

	// compose A
	m_A = SpMatf(nDynVert, nDynVert);
	m_A = m_M + m_dt2DTWeTWeD + m_dt2Wc;
	m_A_damp = m_A + m_Damp_Mat;

	m_I = SpMatf(nDynVert, nDynVert);
	m_I.setIdentity();

	// dual variables
	m_Ue = Matf_X3(m_nCDim, 3);
	m_Ue.setZero();
	m_Uc = Matf_X3(nDynVert, 3);
	m_Uc.setZero();

	// Gamma_c for PGS
	if (enable_frictional_contact && prox_query) {
		Gamma_c.reserve(prox_query->max_collision_num);
	}

	m_LLT_solver = std::make_unique<SolverT>();
	m_LLT_solver->compute(m_A_damp);
	if (m_LLT_solver->info() != Eigen::Success) {
		ADU::make_exception("ADMMSolver::init LLT failed");
	}

	auto save_name_A = "./m_A_damp_" + ADU::getCurTime();
	if (Eigen::saveMarket(m_A_damp, save_name_A)) {
		printf("saved m_A_damp\n");
	}
	else {
		printf("save m_A_damp failed\n");
	}

	CompactSparseMat c(m_A_damp);
	
	// XPBD
	/*m_M_bar_vec = (m_M + m_dt2Wc);
	m_M_bar_inv_vec = Vecf_X(m_nVert);
	for (int i = 0; i < nDynVert; i++) {
		m_M_bar_inv_vec(i) = 1.0_r / m_M_bar_vec.coeff(i, i);
		if (m_pin_inds_set.count(i)) 
			m_M_bar_inv_vec(i) = 0.0_r;
	}*/

	printf("ADMMSolverFull_RL_damping done, total vert num = %d\n", nDynVert);
}

void ADMMSolverFull_RL_damping::step() {
	step_fast();
}

void ADMMSolverFull_RL_damping::step_fast() {
	using namespace ADU;

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

	if (enable_frictional_contact) {}

	// animate_step
	Timer timer("ADMMSolverFull::step()");

#ifdef PROFILE_RR
	x_prev = x_curr;
	rr_list.push_back({});
#endif // PROFILE_RR

	for (int admm_it = 0; admm_it < admm_max_iter; admm_it++) {
		Timer per_iteration_timer("per_iteration");

		// collision
		if (enable_frictional_contact && prox_query && (admm_it % collision_detection_interval == 0)) {
			Timer collision_timer("dynamic_collision_detection");
			//prox_query->proximal_query(x_curr);
			if (use_CCD) {
				#ifdef USE_CCD
				prox_query->proximal_query_with_CCD(x_0, x_curr);
				//prox_query->proximal_query_with_CCD(x_tilde, x_curr);
				#else 
				prox_query->proximal_query(x_curr);
				#endif
			}
			else {
				prox_query->proximal_query(x_curr);
			}

			if (use_unique_contact) {
				prox_query->unique_contact();
			}

			need_recompute_Scc = true;
			find_contact_islands(); // TODO
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
						ADMMSolverFull_RL_damping::project_feasible(p, prox_query->contact_info_list, this->mu, gs_max_iter);
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

			tbb::parallel_invoke(
				[&]() {
					x_curr.block(0, 0, nDynVert, 1) = m_LLT_solver->solve(b.col(0));
				},
				[&]() {
					x_curr.block(0, 1, nDynVert, 1) = m_LLT_solver->solve(b.col(1));
				},
				[&]() {
					x_curr.block(0, 2, nDynVert, 1) = m_LLT_solver->solve(b.col(2));
				}
			);	
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


void ADMMSolverFull_RL_damping::compute_Scc(bool is_XPBD ) {
	using namespace ADU;
	auto& contacts = prox_query->contact_info_list;
	size_t nContact = contacts.size();

	Gamma_c.resize(nContact, Vecf_4::Zero());
	K_c.resize(nContact, Vecf_3::Zero());
	//Real Sc_W{};

	if (!is_XPBD) {
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
					if (contact_w_list(ct.vinds[i]) > 1e7_r) Gamma_c[ci](i) = 0;
					else Gamma_c[ci](i) = ct.bary[i] / (Sc * contact_w_list(ct.vinds[i]));
				}
				
				// precompute k
				// k_c = 1 / dt * h_cN * n
				//K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma * (1.0_r - 2 * epsilon);
				K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma;
			}
			});
	}
	else {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](const tbb::blocked_range<size_t>& r) {
			Real Sc{};
			for (size_t ci = r.begin(); ci < r.end(); ci++) {
				const auto& ct = contacts[ci];
				// TODO: only correct when w_j is all the same
				/*Sc_W = ct.bary.squaredNorm() + epsilon;
				Gamma_c[ci] = ct.bary / Sc_W;*/
				Sc = epsilon;
				for (int i = 0; i < 4; i++) {
					Sc += ct.bary[i] * ct.bary[i] * (m_M_inv_vec(ct.vinds[i]));
				}
				for (int i = 0; i < 4; i++) {
					if (m_M_vec(ct.vinds[i]) > 1e7_r) Gamma_c[ci](i) = 0;
					else Gamma_c[ci](i) = ct.bary[i] / (Sc * m_M_vec(ct.vinds[i]));
				}
				
				// precompute k
				// k_c = 1 / dt * h_cN * n
				K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma;
			}
			});
	}
}

void ADMMSolverFull_RL_damping::project_feasible(ADU::Matf_X3& p,
	ProximalQuery::ContactInfoList& contacts,
	ADU::Real mu, size_t max_GS_iter)
{
	std::cout << "here" << "\n";
	ADMMSolverFull_RL_damping::_project_feasible_plain(p, contacts, mu, max_GS_iter);
}

void ADMMSolverFull_RL_damping::_project_feasible_plain(ADU::Matf_X3& p,
	ProximalQuery::ContactInfoList& contacts,
	ADU::Real mu, size_t max_GS_iter)
{
	// p: current velocity
	// notion: r_c lies in contact_info
	using namespace ADU;

	size_t nContact = contacts.size();

	tbb::parallel_for(tbb::blocked_range<size_t>(0, nContact), [&](const tbb::blocked_range<size_t>& r) {
		for (int ci = r.begin(); ci < r.end(); ci++) {
			const auto& ct = contacts[ci];
			for (size_t k = 0; k < 4; k++) {
				size_t j = ct.vinds[k];
				if (j < static_vert_begin) {
					p.row(j) += Gamma_c[ci](k) * ct.r_c;
				}
			}
		}
		});
	Vecf_3 u{};
	Vecf_3 u_star{};
	Vecf_3 u_star_new{};
	Real u_star_N{};
	Vecf_3 u_star_T{};
	Real tau{};
	Real alpha{};
	for (size_t gs_iter = 0; gs_iter < max_GS_iter; gs_iter++) {
		for (int ci = 0; ci < nContact; ci++) {
			auto& ct = contacts[ci];
			const auto& vinds = ct.vinds;
			u = ct.bary[0] * p.row(vinds[0]) +
				ct.bary[1] * p.row(vinds[1]) +
				ct.bary[2] * p.row(vinds[2]) +
				ct.bary[3] * p.row(vinds[3]);

			u += K_c[ci];
			//u *= (1.0_r - 2 * epsilon);

			u_star = u - ct.r_c;

			// project u_star to satisfy coulomb law
			const auto& normal = ct.normal;
			u_star_N = normal.dot(u_star);
			//u_star_N = normal.dot(u);
			u_star_T = u_star - u_star_N * normal;

			if (u_star_N < 0) {
				if (!use_anisotropy) {
					tau = u_star_T.norm();
					alpha = -mu * u_star_N;
					//u_star_N = - u_star_N * 0.1 ;
					u_star_N = 0;
					if (tau <= alpha) {
						u_star_T = Vecf_3::Zero();
					}
					else {
						u_star_T = (1.0_r - alpha / tau) * u_star_T;
					}
				}
				else {
					ADU::Real mu_t_curr = ct.d * mu_t + (1 - ct.d) * mu;
					ADU::Real mu_b_curr = ct.d * mu_b + (1 - ct.d) * mu;
					const ADU::Vecf_3& t = (ct.Sm - ct.Sm.dot(normal) * normal).normalized();
					const ADU::Vecf_3& b = normal.cross(t).normalized();
					Real tau_t = u_star_T.dot(t);
					Real tau_b = u_star_T.dot(b);
					Real alpha_t = -mu_t_curr * u_star_N;
					Real alpha_b = -mu_b_curr * u_star_N;
					u_star_N = 0;
					u_star_T.setZero();
					if (std::abs(tau_b) > alpha_b) {
						u_star_T += (1.0_r - alpha_b / std::abs(tau_b)) * tau_b * b;
					}
					if (std::abs(tau_t) > alpha_t) {
						u_star_T += (1.0_r - alpha_t / std::abs(tau_t)) * tau_t * t;
					}
				}
			}

			// update force
			u_star_new = u_star_N * ct.normal + u_star_T;
			ct.r_c += u_star_new - u;

			// update P using r_c
			for (size_t k = 0; k < 4; k++) {
				size_t j = ct.vinds[k];
				//p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
				if (j < static_vert_begin) p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
			}
		}
	}	
}



void ADMMSolverFull_RL_damping::find_contact_islands() {
	
}

