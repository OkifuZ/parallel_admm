#include "solver/admm_full_solver.h"
#include "mutils/timer.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>

#include <tbb/parallel_for.h>
#include <mutex>
#include <memory>
#include <random>


void ADMMSolverFull::precompute() {
	ADU::make_exception("ADMMSolverFull::precompute() deprecated");

	using namespace ADU;
	printf("triggered_precompute base\n");

	// construct global matrix
	if (m_constraints.size() == 0) {
		throw std::runtime_error("ADMMSolver::init Error: empty constraints");
	}
	int m = m_constraints.size();

	m_M = SpMatf(m_nVert, m_nVert);
	TripList M_trips;
	for (int vi = 0; vi < m_nVert; vi++) {
		M_trips.emplace_back(vi, vi, m_M_vec(vi));
	}
	m_M.setFromTriplets(M_trips.begin(), M_trips.end());

	// elastic 
	m_dt2DTWeTWeD = SpMatf(m_nVert, m_nVert);
	m_dt2DTWeTWe = SpMatf(m_nVert, m_nCDim);
	m_D = SpMatf(m_nCDim, m_nVert);
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

	// frictional contact
	m_dt2Wc = SpMatf(m_nVert, m_nVert);
	m_W_c = SpMatf(m_nVert, m_nVert);
	if (enable_frictional_contact) {
		TripList Wc_trips;
		for (int i = 0; i < m_nVert; i++) {
			//Wc_trips.emplace_back(i, i, contact_w);
			Wc_trips.emplace_back(i, i, contact_w_list(i));
		}
		m_W_c.setFromTriplets(Wc_trips.begin(), Wc_trips.end());
		m_dt2Wc = m_dt2 * m_W_c;
	}
	else {
		m_dt2Wc.setZero();
	}

	m_A = SpMatf(m_nVert, m_nVert);
	m_A = m_M + m_dt2DTWeTWeD + m_dt2Wc;

	m_LLT_solver = std::make_unique<SolverT>();
	m_LLT_solver->compute(m_A);
	if (m_LLT_solver->info() != Eigen::Success) {
		throw std::runtime_error("ADMMSolver::init Error: LLT failed");
	}

	m_Ue = Matf_X3(m_nCDim, 3);
	m_Ue.setZero();
	m_Uc = Matf_X3(m_nVert, 3);
	m_Uc.setZero();

	if (enable_frictional_contact && prox_query) {
		Gamma_c.reserve(prox_query->max_collision_num);
	}

	// build sparse mat
	saveMarket(m_A, "A");
}

void ADMMSolverFull::init() {
	using namespace ADU;
	Timer timer("ADMMSolverFull::init()");

	this->precompute();
	if (dynamic_cast<ADMMSolverFull_RL_damping*>(this) == nullptr) {
		printf("shit\n");
	}

	for (size_t ni = m_nodal_collision_constraint_start; ni < m_nodal_collision_constraint_end; ni++) {
		std::shared_ptr<NodalCollisionConstraint> ct =
			std::static_pointer_cast<NodalCollisionConstraint>(m_constraints[ni]);
		ct->contact_info = nullptr;
		ct->active = false;
	}

	if (coloring_parallel_contact) {
		v_c_list.resize(m_vertices.rows());
		for (int i = 0; i < v_c_list.size(); i++) {
			v_c_list[i].reserve(64);
		}
		color_result.resize(_color_max_num + 1);
		for (int i = 0; i < _color_max_num; i++) {
			color_result[i].reserve(2000);
		}
	}
	if (animator) {
		animator->reset();
		animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
	}
}


void ADMMSolverFull::reset(const ADU::Matf_X3& ini_verts, bool need_precompute) {
	

	this->m_vertices = ini_verts;
	this->m_velocities.setZero();


	if (warmstart_Ue) { m_Ue.setZero(); }
	if (warmstart_Uc) { m_Uc.setZero(); }

	if (need_precompute) {
		precompute();
	}

	if (coloring_parallel_contact) {
		v_c_list.resize(m_vertices.rows());
		for (int i = 0; i < v_c_list.size(); i++) {
			v_c_list[i].clear();
		}
		color_result.resize(_color_max_num + 1);
		for (int i = 0; i < _color_max_num; i++) {
			color_result[i].clear();
		}
	}

	prox_query->contact_info_list.clear();

	if (animator) {
		animator->reset();
		animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
	}

}


void ADMMSolverFull::project_feasible(ADU::Matf_X3& p, ProximalQuery::ContactInfoList& contacts, ADU::Real mu, size_t max_GS_iter) {
	// p: current velocity
	// notion: r_c lies in contact_info
	using namespace ADU;

	size_t nContact = contacts.size();

	for (int ci = 0; ci < nContact; ci++) {
		const auto& ct = contacts[ci];
		for (size_t k = 0; k < 4; k++) {
			size_t j = ct.vinds[k];
			p.row(j) += Gamma_c[ci](k) * ct.r_c;
		}
	}

	
	if (coloring_parallel_contact) {
		// TODO: remaining part, slower
		for (size_t gs_iter = 0; gs_iter < max_GS_iter; gs_iter++) {
			for (int color = 1; color < color_result.size(); color++) {
				auto& contacts_of_color = color_result[color];
				tbb::task_arena ta(3);
				ta.execute([&]() {
					tbb::parallel_for(tbb::blocked_range<size_t>(0, contacts_of_color.size()),
					[&](tbb::blocked_range<size_t> r) {
							Vecf_3 u{};
							Vecf_3 u_star{};
							Vecf_3 u_star_new{};
							Real u_star_N{};
							Vecf_3 u_star_T{};
							Real tau{};
							Real alpha{};
							for (auto idx = r.begin(); idx < r.end(); idx++) {
								size_t ci = contacts_of_color[idx];
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
								ct.r_c += u_star_new - u;

								// update P using r_c
								for (size_t k = 0; k < 4; k++) {
									size_t j = ct.vinds[k];
									p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
								}
							}
						});
					});
			}
			Vecf_3 u{};
			Vecf_3 u_star{};
			Vecf_3 u_star_new{};
			Real u_star_N{};
			Vecf_3 u_star_T{};
			Real tau{};
			Real alpha{};
			auto& contacts_of_color = color_result[0];
			for (size_t idx = 0; idx < contacts_of_color.size(); idx++) {
				size_t ci = contacts_of_color[idx];
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
				ct.r_c += u_star_new - u;

				// update P using r_c
				for (size_t k = 0; k < 4; k++) {
					size_t j = ct.vinds[k];
					p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
				}
			}
		}
	}
	else {
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

				u_star = u - ct.r_c;

				// project u_star to satisfy coulomb law
				const auto& normal = ct.normal;
				u_star_N = normal.dot(u);
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
				ct.r_c += u_star_new - u;

				// update P using r_c
				for (size_t k = 0; k < 4; k++) {
					size_t j = ct.vinds[k];
					p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
				}
			}
		}
	}
	if (ADMM_VERBOSE) {
		printf("--------------------------------\n");
		printf("r_c:\n");
		for (int ci = 0; ci < nContact; ci++) {
			const auto& ct = contacts[ci];
			printf("%f %f %f\n", ct.r_c.x(), ct.r_c.y(), ct.r_c.z());
		}
	}
}

void ADMMSolverFull::sequential_greedy_coloring(const ProximalQuery::ContactInfoList& cs_list, bool use_rand) {
	for (int i = 0; i <= _color_max_num; i++) {
		color_result[i].clear();
	}
	for (auto& csl : v_c_list) {
		csl.clear();
	}
	for (int ci = 0; ci < cs_list.size(); ci++) {
		const auto& c = cs_list[ci];
		v_c_list[c.vinds[0]].push_back(ci);
		v_c_list[c.vinds[1]].push_back(ci);
		v_c_list[c.vinds[2]].push_back(ci);
		v_c_list[c.vinds[3]].push_back(ci);
	}

	std::mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uni(0, _color_max_num);

	ct_colors = std::vector<int>(cs_list.size(), 0);
	ADU::Veci_X neighbor_color(_color_max_num + 1);
	for (int ci = 0; ci < cs_list.size(); ci++) {
		neighbor_color.setZero();

		// gather neighbor color
		const auto& c = cs_list[ci];
		for (size_t vi : c.vinds) {
			const auto& ncts = v_c_list[vi];
			for (const size_t& nci : ncts) {
				neighbor_color(ct_colors[nci]) = 1;
			}
		}

		// find avalible color
		int start_color{};
		if (use_rand) start_color = uni(rng);
		else start_color = 0;
		int real_c{};
		bool find_one{ false };
		for (int color = start_color; color < start_color + _color_max_num; color++) {
			real_c = color % _color_max_num + 1;
			if (neighbor_color(real_c) == 0) {
				ct_colors[ci] = real_c;
				color_result[real_c].push_back(ci);
				find_one = true;
				break;
			}
		}
		if (!find_one) {
			color_result[0].push_back(ci);
		}
	}

	if (ADMM_VERBOSE) {
		int total = 0;

		printf("color: ");
		for (int i = 0; i <= _color_max_num; i++) {
			printf("%d ", color_result[i].size());
			total += color_result[i].size();
		}
		printf("\n");
		/*max_remaining = std::max(cs_list.size() - total, max_remaining);
		printf("\ntotal: %d, remaining %d, max rm %d\n", total, cs_list.size() - total, max_remaining);*/
	}
}