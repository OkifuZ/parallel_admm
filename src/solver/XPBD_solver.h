#pragma once

#include "solver/solver.h"
#include "solver/admm_full_solver.h"
#include "mutils/timer.h"
#include "mutils/exception_handle.h"
#include "constraint/pin_constraint.h"
#include <mutils/common_type_hostonly.h>



class XPBDSolver {
public:

	Solver* solver;
	std::vector<std::vector<ADU::Real>> x_residual;
	std::vector<std::vector<ADU::Real>> time_spans;

	unsigned int m_subSteps = 1;
	unsigned int m_maxIterations = 400;
	ADU::Real contact_stiffness = 10;
	bool use_GS_contact = true;

	ADU::Matf_X3 x_prev_rr;
	std::vector<std::vector<ADU::Real>> rr_list;

	virtual void init(Solver* solver, int max_iter, int substep) {
		using namespace ADU;
		Timer timer("XPBDSolver::init()");
		this->solver = solver;
		if (!solver) make_exception("XPBD empty ADMM solver");
		ADMMSolverFull_RL_damping* admm_solver = reinterpret_cast<ADMMSolverFull_RL_damping*>(solver);
		this->m_maxIterations = max_iter;
		this->m_subSteps = substep;

		int nDynVert = admm_solver->static_vert_begin;
#ifdef PROFILE_RR
		x_prev_rr.resize(nDynVert, 3);
#endif // PROFILE_RR

	}

	virtual void step() {
		step_XPBD();
	}

	void step_XPBD() {
		printf("step XPBD\n");



		using namespace ADU;
		if (!solver) make_exception("XPBD empty ADMM solver");
		ADMMSolverFull_RL_damping* admm_solver = reinterpret_cast<ADMMSolverFull_RL_damping*>(solver);
		using namespace ADU;
		Timer timer("XPBDSolver::step()");

		Real h = admm_solver->m_dt / m_subSteps;
		Real h_inv = 1.0_r / h;

		for (auto& ct : admm_solver->m_constraints) {
			if (ct->type == 4) {
				auto& pct = std::static_pointer_cast<PinConstraint>(ct);
				admm_solver->m_vertices.row(ct->inds[0]) = pct->pin_poistion;
			}
		}
		int nDynVert = admm_solver->static_vert_begin;
		Matf_X3 x_prev(admm_solver->m_vertices);
		Matf_X3 x_prev_iter(admm_solver->m_vertices);
		Matf_X3 x_curr(admm_solver->m_vertices);
		Matf_X3 v_curr(admm_solver->m_velocities);

#ifdef PROFILE_RR
		if (m_subSteps != 1) printf("!!!!!!!!!!!!!!!!!!error!!!!!!!!!!!!!! m_substep != 1\n");
		rr_list.push_back({});
#endif // PROFILE_RR

		x_residual.push_back({});
		time_spans.push_back({});

		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

		for (unsigned int substep = 0; substep < m_subSteps; substep++) {
			x_prev = x_curr;
			for (int i = 0; i < nDynVert; i++) {
				if (admm_solver->m_pin_inds_set.count(i)) {
					continue;
				}
				v_curr.row(i).y() -= admm_solver->g * h;
			}
			x_curr += v_curr * h;

			x_prev_iter = x_curr;

			int cur_iter = 0;
			while (cur_iter < m_maxIterations) {
				if (admm_solver->enable_frictional_contact && admm_solver->prox_query && cur_iter % admm_solver->collision_detection_interval == 0) {
					Timer collision_timer("dynamic_collision_detection");
					//prox_query->proximal_query(x_curr);
					if (admm_solver->use_CCD) {
						#ifdef USE_CCD
						admm_solver->prox_query->proximal_query_with_CCD(x_0, x_curr);
						//prox_query->proximal_query_with_CCD(x_tilde, x_curr);
						#else 
						admm_solver->prox_query->proximal_query(x_curr);
						#endif
					}
					else {
						admm_solver->prox_query->proximal_query(x_curr);
					}

					if (admm_solver->use_unique_contact) {
						admm_solver->prox_query->unique_contact();
					}

					admm_solver->need_recompute_Scc = true;
				}

				for (unsigned int ci = 0; ci< admm_solver->m_XPBDconstraints.size(); ci++)
				{
					admm_solver->m_XPBDconstraints[ci]->updateConstraint();
					admm_solver->m_XPBDconstraints[ci]->solvePositionConstraint(x_curr, Matf_X3{}, admm_solver->m_M_inv_vec, cur_iter, h);
				}

				// solve velocity
				if (admm_solver->enable_frictional_contact)
				{ // frictional contact
					Timer local_project_timer("contact_local");

					if (use_GS_contact) {

						v_curr = h_inv * (x_curr - x_prev);

						if (admm_solver->need_recompute_Scc) {
							admm_solver->compute_Scc(true);
							admm_solver->need_recompute_Scc = false;
						}

						// project p into feasible set
						admm_solver->project_feasible(v_curr, admm_solver->prox_query->contact_info_list, admm_solver->mu, admm_solver->gs_max_iter);
						x_curr = x_prev + h * v_curr;
					}
					else {
						for (int i = 0; i < 5; i++) {
							// just solve contact constraints
							solve_contact_constraints(admm_solver->prox_query->contact_info_list, admm_solver->m_M_inv_vec, x_curr);
						}
					}
				}
				//x_residual.back().push_back((x_curr - x_prev_iter).squaredNorm());
#ifdef PROFILE_RR
				Real rr = (x_curr - x_prev_iter).norm() / x_curr.norm();
				rr_list.back().push_back(rr);
#endif // PROFILE_RR
				x_prev_iter = x_curr;
				cur_iter++;

				/*std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
				std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
				time_spans.back().push_back(time_span.count());*/
			}

			v_curr = h_inv * (x_curr - x_prev);
			v_curr *= 0.9995;
		}
		admm_solver->m_velocities = v_curr;
		admm_solver->m_vertices = x_curr;
	}

	void solve_contact_constraints(const ProximalQuery::ContactInfoList& contacts, const ADU::Vecf_X& M_inv_list, 
		ADU::Matf_X3& pos) {
		using namespace ADU;
		std::array<Vecf_3, 4> corr;
		Real compress_stiff = this->contact_stiffness;
		Real alpha = 1.0_r;

		for (int i = 0; i < contacts.size(); i++) {
			auto& ct = contacts[i];
			const auto& inds = ct.vinds;

			Real m_inv0 = M_inv_list[inds[0]];
			Real m_inv1 = M_inv_list[inds[1]];
			Real m_inv2 = M_inv_list[inds[2]];
			Real m_inv3 = M_inv_list[inds[3]];

			bool res{ false };
			if (ct.pair_type == ContactInfo::PT) {
				res = XPBD_utils::solve_TrianglePointDistanceConstraint(
					pos.row(inds[0]), m_inv0,
					pos.row(inds[1]), m_inv1,
					pos.row(inds[2]), m_inv2,
					pos.row(inds[3]), m_inv3,
					ContactParameter::get_thickness(),
					compress_stiff, 0,
					corr[0], corr[1], corr[2], corr[3]
				);
			}
			else if (ct.pair_type == ContactInfo::EE) {
				res = XPBD_utils::solve_EdgeEdgeDistanceConstraint(
					pos.row(inds[0]), m_inv0,
					pos.row(inds[1]), m_inv1,
					pos.row(inds[2]), m_inv2,
					pos.row(inds[3]), m_inv3,
					ContactParameter::get_thickness(),
					compress_stiff, 0,
					corr[0], corr[1], corr[2], corr[3]
				);
			}
			else {}

			if (res) {
				pos.row(inds[0]) += corr[0] * alpha;
				pos.row(inds[1]) += corr[1] * alpha;
				pos.row(inds[2]) += corr[2] * alpha;
				pos.row(inds[3]) += corr[3] * alpha;
			}
		}

	}

	virtual void reset(const ADU::Matf_X3& ini_verts, bool need_precompute = false) {
		using namespace ADU;
		if (!solver) make_exception("XPBD empty ADMM solver");
		ADMMSolverFull_RL_damping* admm_solver = reinterpret_cast<ADMMSolverFull_RL_damping*>(solver);
		admm_solver->m_vertices = ini_verts;
		admm_solver->m_velocities.setZero();
	}



};
