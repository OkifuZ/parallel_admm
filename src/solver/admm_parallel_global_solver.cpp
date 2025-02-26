#include "solver/admm_full_solver.h"
#include "mutils/timer.h"
#include "constraint/pin_constraint.h"
#include "thrust/device_vector.h"
#include "solver/contact_solver.cuh"


#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <mutex>
#include <memory>
#include <random>


void ADMMParallelSolver::convert_constraint2device() {
	// 1. count constraint num for each type
	std::vector<std::shared_ptr<TriangleConstraint>> tri_constraints;
	std::vector<std::shared_ptr<PinConstraint> > pin_constraints;

	for (auto& ct: m_constraints) {
		if (ct->type == 1) {
			static bool hit1 = false;
			if (!hit1) {
				hit1 = true;
				triangle_constraint_start_row = ct->start_row;
			}
			tri_constraints.push_back(std::dynamic_pointer_cast<TriangleConstraint>(ct));
		}
		else if (ct->type == 2) {
		}
		else if (ct->type == 3) {
		}
		else if (ct->type == 4) {
			static bool hit4 = false;
			if (!hit4) {
				hit4 = true;
				pin_constraint_start_row = ct->start_row;
			}
			pin_constraints.push_back(std::dynamic_pointer_cast<PinConstraint>(ct));
		}
		else if (ct->type == 5) {
		}
		else if (ct->type == 6) {
		}
	}

	std::cout << triangle_constraint_start_row << std::endl;
	std::cout << pin_constraint_start_row << std::endl;

	this->pin_constraint_cu = std::make_unique<PinConstraintDevice>(pin_constraints);
	this->triangle_constraint_cu = std::make_unique<TriangleConstraintDevice>(tri_constraints);
}



void ADMMParallelSolver::init() {
	printf("ADMMParallelSolver::init start\n");


	precompute();

	if (animator) {
		animator->reset();
		animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
	}
	int nDynVert = static_vert_begin;

	comp_mat = std::make_unique<CompactSparseMat>(m_A_damp); // for parallel global solver
	solver_data_device = std::make_unique<CuSolverData>(*comp_mat,
		nDynVert, m_nVert, nDynVert, m_nVert, m_nCDim);
	// comp_mat_device = std::make_unique<CuCompactSparseMat>(*comp_mat);

    jacobi_buffer.resize(nDynVert, 3);
	jacobi_buffer.setZero();

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

	resizeThrust<ADU::Real>(this->cache_nCDimX3, m_nCDim * 3, 0);
	resizeThrust<ADU::Real>(this->cache_nDynVertX3, nDynVert * 3, 0);
	resizeThrust<ADU::Real>(this->cache_nDynVertX3_bp1, nDynVert*3, 0);


	printf("ADMMParallelSolver init done\n");

}


void ADMMParallelSolver::precompute() {
	using namespace ADU;

	printf("ADMMParallelSolver::precompute start\n");


	// construct global matrix
	if (m_constraints.size() == 0) {
		throw std::runtime_error("ADMMParallelSolver::precompute Error: empty constraints");
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

	m_M_device = std::make_unique<CompactSparseMat>(m_M, false);

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
	m_D_device = std::make_unique<CompactSparseMat>(m_D, false);

	m_dt2DTWeTWe = m_dt2 * m_D.transpose() * m_W_e.transpose() * m_W_e;
	m_dt2DTWeTWeD = m_dt2DTWeTWe * m_D;
	m_dt2DTWeTWe_device = std::make_unique<CompactSparseMat>(m_dt2DTWeTWe, false);

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

	m_dt2Wc_device = std::make_unique<CompactSparseMat>(m_dt2Wc, false);


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

	resizeThrust<ADU::Real>(m_Ue_device, 3*m_nCDim, 0);
	resizeThrust<ADU::Real>(m_Uc_device, 3 * nDynVert, 0);

	// Gamma_c for PGS
	if (enable_frictional_contact && prox_query) {
		Gamma_c.reserve(prox_query->max_collision_num);
	}

	m_LLT_solver = std::make_unique<SolverT>();
	m_LLT_solver->compute(m_A_damp);
	if (m_LLT_solver->info() != Eigen::Success) {
		ADU::make_exception("ADMMParallelSolver::init LLT failed");
	}

	auto save_name_A = "./m_A_damp_" + ADU::getCurTime();
	if (Eigen::saveMarket(m_A_damp, save_name_A)) {
		printf("ADMMParallelSolver: saved m_A_damp\n");
	}
	else {
		printf("ADMMParallelSolver: save m_A_damp failed\n");
	}
	CompactSparseMat c(m_A_damp);

	printf("ADMMParallelSolver done, total vert num = %d\n", nDynVert);


	resizeThrust<int>(d_vi_ct_nums, m_nVert, 0);
}



void ADMMParallelSolver::step() {
    using namespace ADU;

	printf("here\n");

	// for contact stabilization
	epsilon = 1.0_r / (m_dt2 * kappa + m_dt * beta);
	gamma = m_dt * kappa / (m_dt * kappa + beta);

	int nDynVert = static_vert_begin;

	/*Matf_X3 b = Matf_X3(nDynVert, 3);
	b.setZero();*/

	Matf_X3 x_0(m_nVert, 3);
	Matf_X3 v_0(m_nVert, 3);
	x_0 = m_vertices;
	v_0 = m_velocities;

	copy_mat2thrustvector(x_0, solver_data_device->x_0_device, m_nVert);

	// explicit integrate as initial value
	for (int i = 0; i < nDynVert; i++) {
		if (m_pin_inds_set.count(i)) v_0.row(i).setZero();
		else v_0.row(i).y() -= g * m_dt;
	}

	ADU::Matf_X3 temp(m_nCDim, 3);

	Matf_X3 x_tilde = x_0.block(0, 0, nDynVert, 3) + m_dt * v_0.block(0, 0, nDynVert, 3); // nDynVert * 3
	Matf_X3 M_x_tilde = m_M * x_tilde; // nDynVert * 3

	resizeThrust<ADU::Real>(M_x_tilde_device, nDynVert * 3, 0);
	copy_mat2thrustvector(M_x_tilde, M_x_tilde_device, nDynVert);

	b_curr.resize(nDynVert, 3);
	b_curr.setZero();

	// initial value
	x_curr.resize(m_nVert, 3);
	x_curr.setZero();
	// Matf_X3 x_curr(m_nVert, 3);
	x_curr.block(0, 0, nDynVert, 3) = x_tilde;
	x_curr.block(nDynVert, 0, m_nVert - nDynVert, 3) = x_0.block(nDynVert, 0, m_nVert - nDynVert, 3);
	Matf_X3 z(m_nCDim, 3);
	z.setZero();
	Matf_X3 DX(m_nCDim, 3);
	DX.setZero();
	resizeThrust<ADU::Real>(DX_device, m_nCDim * 3, 0);

	if (animator) {
		animator->animate_all(m_vertices, x_curr, v_0, this->m_dt);
	}
	Matf_X3 b_ini = m_Damp_Mat * x_0.block(0, 0, nDynVert, 3);

	p.resize(m_nVert, 3);
	p = v_0;

	if (!warmstart_Ue) { m_Ue.setZero(); }
	if (!warmstart_Uc) { m_Uc.setZero(); }

	Timer timer("ADMMParallelSolver::step()");

	copy_mat2thrustvector(x_curr, solver_data_device->x_curr_device, m_nVert);

	for (int admm_it = 0; admm_it < admm_max_iter; admm_it++) {
		Timer per_iteration_timer("per_iteration");


		// collision
		if (enable_frictional_contact && prox_query && (admm_it % collision_detection_interval == 0)) {
			Timer collision_timer("dynamic_collision_detection");

			// bvh->update(x_curr, true);
			bvh->update(solver_data_device->x_curr_device.data().get(), m_nVert, true);
			bvh->dcd();

			bvh->unique_contactInfo();
			bvh->convert_contactInfo_device2host(prox_query->contact_info_list, x_curr);

			need_recompute_Scc = true;

			// find_contact_islands(); // TODO
		}

		{
			Timer local_project_timer("elastic_local & contact_local");


			{ // elasticity
				/*DX = m_D * x_curr.block(0, 0, nDynVert, 3); // computed after global, since x_curr never changes between two admm iterations
				z = DX + m_Ue;*/
				op_Ax(*m_D_device, solver_data_device->x_curr_device, DX_device);
				op_a_plus_b(DX_device, m_Ue_device, solver_data_device->z_buffer, m_nCDim);


				/*copy_thrustvector2mat(solver_data_device->z_buffer, z, m_nCDim);

				tbb::parallel_for_each(m_constraints.begin(), m_constraints.end(), [&](const std::shared_ptr<Constraint>& ct) {
					int cdim = ct->dim;
					Matf_XX zi = z.block(ct->start_row, 0, cdim, 3);
					ct->prox(zi);
					z.block(ct->start_row, 0, cdim, 3) = zi;
					});

				copy_mat2thrustvector(z, solver_data_device->z_buffer, m_nCDim);*/

				// TODO device ptr?
				triangle_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + triangle_constraint_start_row * 3));

				pin_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + pin_constraint_start_row * 3));
			}

			{
				if (enable_frictional_contact)
				{ // frictional contact
					//Timer local_project_timer("contact_local");

					/*p.block(0, 0, nDynVert, 3) =
						(x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) + m_Uc) * m_dt_inv; // p as start velocity
						*/
					op_a_minus_b(solver_data_device->x_curr_device, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
					op_a_plus_b(cache_nDynVertX3, m_Uc_device, cache_nDynVertX3, nDynVert);
					op_scale(cache_nDynVertX3, m_dt_inv, solver_data_device->p_device, nDynVert);

					if (need_recompute_Scc) {
						ADMMSolverFull_RL_damping::compute_Scc();
						need_recompute_Scc = false;
					}

					/*if (need_recompute_Scc) {
						ADMMSolverFull_RL_damping::compute_Scc();
						need_recompute_Scc = false;
					}
					ADMMSolverFull_RL_damping::project_feasible(p, prox_query->contact_info_list, this->mu, gs_max_iter);*/
					
				}
			}
		}

		{
			Timer local_project_timer("global_propogation");

			// global
			op_a_minus_b(solver_data_device->z_buffer, m_Ue_device, cache_nCDimX3, m_nCDim);
			op_Ax(*m_dt2DTWeTWe_device, cache_nCDimX3, cache_nDynVertX3);
			op_a_plus_b(M_x_tilde_device, cache_nDynVertX3, solver_data_device->b_curr_device, nDynVert);
			//b_curr = M_x_tilde + m_dt2DTWeTWe * (z - m_Ue);

			// friction
			if (enable_frictional_contact) {
				op_scale(solver_data_device->p_device, m_dt, cache_nDynVertX3, nDynVert);
				op_a_plus_b(cache_nDynVertX3, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
				op_a_minus_b(cache_nDynVertX3, m_Uc_device, cache_nDynVertX3, nDynVert);
				op_Ax(*m_dt2Wc_device, cache_nDynVertX3, cache_nDynVertX3);
				op_a_plus_b(solver_data_device->b_curr_device, cache_nDynVertX3, solver_data_device->b_curr_device, nDynVert);
				//b_curr += m_dt2Wc * (m_dt * p.block(0, 0, nDynVert, 3) + x_0.block(0, 0, nDynVert, 3) - m_Uc);
			}
			// damp
			//b_curr += b_ini;

            // GS_global(b_curr, x_curr, 45);
            Jacobi_global(20);

			/*copy_thrustvector2mat(solver_data_device->x_curr_device, x_curr, nDynVert);
			copy_thrustvector2mat(solver_data_device->b_curr_device, b, nDynVert);
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
			copy_mat2thrustvector(x_curr, solver_data_device->x_curr_device, nDynVert);*/
		}

		op_Ax(*m_D_device, solver_data_device->x_curr_device, DX_device);
		op_a_minus_b(DX_device, solver_data_device->z_buffer, cache_nCDimX3, m_nCDim);
		op_a_plus_b(cache_nCDimX3, m_Ue_device, m_Ue_device, m_nCDim);
		//DX = m_D * x_curr.block(0, 0, nDynVert, 3);
		//m_Ue += DX - z;

		if (enable_frictional_contact) {
			op_a_minus_b(solver_data_device->x_curr_device, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
			op_scale(solver_data_device->p_device, m_dt, cache_nDynVertX3_bp1, nDynVert);
			op_a_minus_b(cache_nDynVertX3, cache_nDynVertX3_bp1, cache_nDynVertX3, nDynVert);
			op_a_plus_b(m_Uc_device, cache_nDynVertX3, m_Uc_device, nDynVert);
			// m_Uc += x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) - p.block(0, 0, nDynVert, 3) * m_dt;
		}
	}

	copy_thrustvector2mat(solver_data_device->x_curr_device,x_curr,  m_nVert);

	m_velocities.block(0, 0, nDynVert, 3) = (x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3)) / m_dt;
	m_velocities.block(nDynVert, 0, m_nVert - nDynVert, 3) = v_0.block(nDynVert, 0, m_nVert - nDynVert, 3);
	m_vertices = x_curr;

	step_cnt++;
}

void ADMMParallelSolver::GS_global(const ADU::Matf_X3& b, ADU::Matf_X3& x_curr, int iter_cnt) {
    using namespace ADU;

    int nDynVert = static_vert_begin;

	for (int i = 0; i < iter_cnt; i++) {
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

}

//void ADMMParallelSolver::Jacobi_global(int iter_cnt, bool on_device) {
//    using namespace ADU;
//
//    int nDynVert = static_vert_begin;
//
//	if (on_device) {
//
//		// copy host to device (temp)
//		//copy_mat2thrustvector(x_curr, solver_data_device->x_curr_device, m_nVert);
//		//copy_mat2thrustvector(b_curr, solver_data_device->b_curr_device, m_nVert);
//
//		cu_jacobi_global(solver_data_device->sp_mat_device,
//			solver_data_device->b_curr_device,
//			solver_data_device->x_curr_device,
//			solver_data_device->jacobi_buffer_1,solver_data_device->jacobi_buffer_2, nDynVert, iter_cnt);
//
//		// copy device to host (temp)
//		//copy_thrustvector2mat(solver_data_device->x_curr_device,x_curr,  m_nVert);
//
//	}
//	else {
//		for (int i= 0; i < iter_cnt; i++) {
//			tbb::parallel_for(
//				tbb::blocked_range<size_t>(0, nDynVert),
//				[&](const tbb::blocked_range<size_t>& r) {
//					for (size_t i = r.begin(); i < r.end(); i++) {
//						Real a_ii = comp_mat->diag_data[i];
//						Vecf_3 b_i_p = b.row(i);
//
//						Vecf_3 b_i_m = Vecf_3::Zero();
//						int st_ind = comp_mat->offdiag_perline_start[i];
//						int ed_ind = comp_mat->offdiag_perline_start[i+1];
//						for (int j = st_ind; j < ed_ind; j++) {
//							int ind = comp_mat->offdiag_indices[j]; // col
//							b_i_m += x_curr.row(ind) * comp_mat->offdiag_data[j];
//						}
//
//						jacobi_buffer.row(i) = (b_i_p - b_i_m) / a_ii;
//					}
//			});
//			x_curr.block(0, 0, nDynVert, 3) = jacobi_buffer;
//		}
//
//	}
//}

void ADMMParallelSolver::Jacobi_global(int iter_cnt) {
	using namespace ADU;

	int nDynVert = static_vert_begin;

	// copy host to device (temp)
	//copy_mat2thrustvector(x_curr, solver_data_device->x_curr_device, m_nVert);
	//copy_mat2thrustvector(b_curr, solver_data_device->b_curr_device, m_nVert);

	cu_jacobi_global(solver_data_device->sp_mat_device,
		solver_data_device->b_curr_device,
		solver_data_device->x_curr_device,
		solver_data_device->jacobi_buffer_1, solver_data_device->jacobi_buffer_2, nDynVert, iter_cnt);

	// copy device to host (temp)
	//copy_thrustvector2mat(solver_data_device->x_curr_device,x_curr,  m_nVert);

}

void ADMMParallelSolver::project_feasible(ADU::Matf_X3& p,
	ProximalQuery::ContactInfoList& contacts,
	ADU::Real mu, size_t max_GS_iter)
{
	ADMMParallelSolver::_project_feasible_plain(p, contacts, mu, max_GS_iter);
	//ADMMParallelSolver::_project_feasible_plain_2(p, contacts, mu, max_GS_iter);
}

void ADMMParallelSolver::_project_feasible_plain(ADU::Matf_X3& p,
		ProximalQuery::ContactInfoList& contacts,
		ADU::Real mu, size_t max_iter)
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


	// pure Jacobi
	for (size_t gs_iter = 0; gs_iter < max_iter; gs_iter++) {
		tbb::task_arena task_arena = tbb::task_arena(16);
		task_arena.execute([&]() {
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
				//u_star_N = normal.dot(u);
				u_star_N = normal.dot(u_star);
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

				/*
				// report
				for (size_t k = 0; k < 4; k++) {
					size_t j = ct.vinds[k];
					//p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
					if (j < static_vert_begin) p.row(j) += Gamma_c[ci](k) * (u_star_new - u);
				}*/
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
		});


	}
}



// input: contacts
// output: Gamma_c, Gamma_i, involved_cid, K_c, delta_u
void ADMMParallelSolver::compute_Scc(bool is_XPBD) {
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
				size_t vi = ct.vinds[i];
				Sc += ct.bary[i] * ct.bary[i] * contact_w_inv_list(vi) * vi_ct_nums[vi];
			}
			for (int i = 0; i < 4; i++) {
				int vi = ct.vinds[i];
				if (contact_w_list(ct.vinds[i]) > 1e7_r) {
					Gamma_c[ci](i) = 0;
				}
				else {
					Gamma_c[ci](i) = ct.bary[i] / (Sc * contact_w_list(vi));
					Gamma_i[vi].emplace_back(ct.bary[i] / (Sc * contact_w_list(vi)));
				}
				involved_cid[vi].push_back(ci);

			}
			
			// precompute k
			// k_c = 1 / dt * h_cN * n
			K_c[ci] = this->m_dt_inv * ct.h_cN * ct.normal * gamma;
		}
		});

}


void ADMMParallelSolver::compute_Scc_impl() {
	using namespace ADU;
	size_t nContact = bvh->h_cpNum;

	//d_Gamma_c.resize(nContact, Vecf_4::Zero());
	////Gamma_c.resize(nContact, Vecf_4::Zero());
	////K_c.resize(nContact, Vecf_3::Zero());
	//d_K_c.resize(nContact, Vecf_3::Zero());
	//Real Sc_W{};

	resizeThrust<float4>(d_Gamma_c, nContact, float4{ 0, 0, 0, 0 });
	resizeThrust<float3>(d_K_c, nContact, float3{ 0, 0, 0 });


	for (int vi = 0; vi < m_nVert; vi++) {
		Gamma_i[vi].clear();
		involved_cid[vi].clear();
	}
	std::fill(vi_ct_nums.begin(), vi_ct_nums.end(), 0);
	delta_u.resize(nContact, Vecf_3::Zero());

	computeContactCountsWithAtomic(bvh->d_collisonPairs, d_vi_ct_nums, nContact);




}