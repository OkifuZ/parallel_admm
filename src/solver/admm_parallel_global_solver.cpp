#include "solver/parallel_solver.h"
#include "mutils/timer.h"
#include "constraint/pin_constraint.h"
#include "constraint/bending_constraint.h"
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
	std::vector<std::shared_ptr<BendingConstraint> > bend_constraints;
	std::vector<std::shared_ptr<TetrahedralConstraint> > tet_constraints;

	for (auto& ct: m_constraints) {
		if (ct->type == 1) {
			static bool hit1 = false;
			if (!hit1) {
				hit1 = true;
				triangle_constraint_start_row = ct->start_row;
			}
			tri_constraints.push_back(std::dynamic_pointer_cast<TriangleConstraint>(ct));
			//printf("tri w %f; ", ct->w);

		}
		else if (ct->type == 2) {
			static bool hit2 = false;
			if (!hit2) {
				hit2 = true;
				tet_constraint_start_row = ct->start_row;
			}
			tet_constraints.push_back(std::dynamic_pointer_cast<TetrahedralConstraint>(ct));
		}
		else if (ct->type == 3) {
			static bool hit3 = false;
			if (!hit3) {
				hit3 = true;
				bending_constraint_start_row = ct->start_row;
			}
			bend_constraints.push_back(std::dynamic_pointer_cast<BendingConstraint>(ct));
			/*for (auto x : bend_constraints.back()->coeffs) {
				printf("%f ", x);
			}
			printf("\n");*/
			//printf("bend w %f; ", ct->w);

		}
		else if (ct->type == 4) {
			static bool hit4 = false;
			if (!hit4) {
				hit4 = true;
				pin_constraint_start_row = ct->start_row;
			}
			pin_constraints.push_back(std::dynamic_pointer_cast<PinConstraint>(ct));
			//printf("pin w %f; ", ct->w);
		}
		else if (ct->type == 5) {
		}
		else if (ct->type == 6) {
		}
	}

	std::cout << "tri sr: " << triangle_constraint_start_row << std::endl;
	std::cout << "pin sr: " << pin_constraint_start_row << std::endl;
	std::cout << "bend sr: " << bending_constraint_start_row << std::endl;
	std::cout << "tet sr: " << tet_constraint_start_row << std::endl;

	this->pin_constraint_cu = std::make_unique<PinConstraintDevice>(pin_constraints);
	this->triangle_constraint_cu = std::make_unique<TriangleConstraintDevice>(tri_constraints);
	this->bending_constraint_cu = std::make_unique<BendingConstraintDevice>(bend_constraints);
	this->tet_constraint_cu = std::make_unique<TetrahedralConstraintDevice>(tet_constraints);
}



void ADMMParallelSolver::init() {
	printf("ADMMParallelSolver::init start\n");

	contact_normals.resize(prox_query->max_collision_num);
	contact_points.resize(prox_query->max_collision_num);


	ct_data_device = std::make_shared<ContactDataDevice>(m_nVert, prox_query->max_collision_num);

	precompute();

	if (animator) {
		animator->reset();
		animator->animate_all(m_vertices, m_vertices, m_velocities, this->m_dt);
	}
	int nDynVert = static_vert_begin;

	//comp_mat = std::make_unique<CompactSparseMat>(m_A_damp); // for parallel global solver
	comp_mat = std::make_unique<CompactSparseMat>(m_A); // for parallel global solver
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

	// for pre-integration
	std::vector<int> is_fixed(m_nVert, 0);
	for (auto x : m_pin_inds_set) {
		is_fixed[x] = 1;
	}
	resizeThrust(d_is_fixed, m_nVert, 0);
	thrust::copy(is_fixed.begin(), is_fixed.end(), d_is_fixed.begin());

	resizeThrust(d_M, m_nVert, 1000000.0f);
	copy_vec2thrustvector(m_M_vec, d_M, m_nVert);

	resizeThrust<ADU::Real>(M_x_tilde_device, m_nVert * 3, 0);
	resizeThrust<ADU::Real>(DX_device, m_nCDim * 3, 0);

	ADU::Matf_X3& x_0 = m_vertices;
	ADU::Matf_X3& v_0 = m_velocities;

	copy_mat2thrustvector(x_0, solver_data_device->x_0_device, m_nVert);
	copy_mat2thrustvector(v_0, solver_data_device->v_0_device, m_nVert);

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
	printf("m_D_device: c %d, r %d\n", m_D_device->cols, m_D_device->rows);

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
	if (enable_frictional_contact) {
		m_A = m_M + m_dt2DTWeTWeD + m_dt2Wc;
	}
	else {
		m_A = m_M + m_dt2DTWeTWeD;
	}
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


	resizeThrust<int>(ct_data_device->d_vi_ct_nums, m_nVert, 0);
	resizeThrust<ADU::Real>(ct_data_device->d_contact_W_list, m_nVert, 0);
	resizeThrust<ADU::Real>(ct_data_device->d_contact_W_inv_list, m_nVert, 0);
	copy_vec2thrustvector(contact_w_list, ct_data_device->d_contact_W_list, m_nVert);
	copy_vec2thrustvector(contact_w_inv_list, ct_data_device->d_contact_W_inv_list, m_nVert);
}



void ADMMParallelSolver::step() {
    using namespace ADU;

	// for contact stabilization
	epsilon = 1.0_r / (m_dt2 * kappa + m_dt * beta);
	gamma = m_dt * kappa / (m_dt * kappa + beta);

	int nDynVert = static_vert_begin;

	x_curr.resize(m_nVert, 3);
	x_curr.setZero();
	b_curr.resize(nDynVert, 3);
	b_curr.setZero();

	do_pre_integration(m_nVert, nDynVert, m_dt, g, solver_data_device->x_0_device.data().get(), 
		d_is_fixed.data().get(), d_M.data().get(),  solver_data_device->v_0_device.data().get(),
		solver_data_device->x_curr_device.data().get(), M_x_tilde_device.data().get());

	if (!warmstart_Ue) { m_Ue.setZero(); }
	if (!warmstart_Uc) { m_Uc.setZero(); }

	Timer timer("ADMMParallelSolver::step()");

	float dt_r = m_dt ;
	float dt_r_inv = 1.0f / dt_r;


	for (int admm_it = 0; admm_it < admm_max_iter; admm_it++) {
		Timer per_iteration_timer("per_iteration");

		// collision
		if (enable_frictional_contact && prox_query && (admm_it % collision_detection_interval == 0)) {
			Timer collision_timer("dynamic_collision_detection");
			bvh->update(solver_data_device->x_curr_device.data().get(), m_nVert, true);
			bvh->dcd();
			// TODO: will this actually work?
			convert_DCD_info(bvh, solver_data_device->x_curr_device, ct_data_device.get());

			need_recompute_Scc = true;
		}

		{
			Timer local_project_timer("elastic_local & contact_local");


			{ // elasticity
				/*DX = m_D * x_curr.block(0, 0, nDynVert, 3); // computed after global, since x_curr never changes between two admm iterations
				z = DX + m_Ue;*/
				op_Ax(*m_D_device, solver_data_device->x_curr_device, DX_device);
				op_a_plus_b(DX_device, m_Ue_device, solver_data_device->z_buffer, m_nCDim);

				// TODO device ptr?
				tet_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + tet_constraint_start_row * 3));
				triangle_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + triangle_constraint_start_row * 3));
				bending_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + bending_constraint_start_row * 3) );
				pin_constraint_cu->run_proxy(thrust::raw_pointer_cast(solver_data_device->z_buffer.data() + pin_constraint_start_row * 3));
			}

			if (enable_frictional_contact)
			{ // frictional contact
				//Timer local_project_timer("contact_local");

				/*p.block(0, 0, nDynVert, 3) =
					(x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) + m_Uc) * m_dt_inv; // p as start velocity
					*/
				op_a_minus_b(solver_data_device->x_curr_device, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
				op_a_plus_b(cache_nDynVertX3, m_Uc_device, cache_nDynVertX3, nDynVert);
				op_scale(cache_nDynVertX3, dt_r_inv, solver_data_device->p_device, nDynVert);

				if (need_recompute_Scc) {
					//ADMMSolverFull_RL_damping::compute_Scc();
					//ADMMParallelSolver::compute_Scc();
					ADMMParallelSolver::compute_Scc_parallel();
					need_recompute_Scc = false;
				}
					ADMMParallelSolver::project_feasible_parallel();

			}
		}

		{
			Timer local_project_timer("global_propogation");

			// global
			//b_curr = M_x_tilde + m_dt2DTWeTWe * (z - m_Ue);
			op_a_minus_b(solver_data_device->z_buffer, m_Ue_device, cache_nCDimX3, m_nCDim);
			op_Ax(*m_dt2DTWeTWe_device, cache_nCDimX3, cache_nDynVertX3);
			op_a_plus_b(M_x_tilde_device, cache_nDynVertX3, solver_data_device->b_curr_device, nDynVert);
			//op_a_to_b(M_x_tilde_device, solver_data_device->b_curr_device, m_nVert);

			// friction
			if (enable_frictional_contact) {
				//b_curr += m_dt2Wc * (m_dt * p.block(0, 0, nDynVert, 3) + x_0.block(0, 0, nDynVert, 3) - m_Uc);
				op_scale(solver_data_device->p_device, dt_r, cache_nDynVertX3, nDynVert);
				op_a_plus_b(cache_nDynVertX3, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
				op_a_minus_b(cache_nDynVertX3, m_Uc_device, cache_nDynVertX3, nDynVert);
				op_Ax(*m_dt2Wc_device, cache_nDynVertX3, cache_nDynVertX3);
				op_a_plus_b(solver_data_device->b_curr_device, cache_nDynVertX3, solver_data_device->b_curr_device, nDynVert);
			}
			// damp
			//b_curr += b_ini;
			// TODO
			//printf("jac iter: %d \n", Global_Jacobi_iter);
            Jacobi_global(Global_Jacobi_iter);
			// copy_thrustvector2mat(solver_data_device->x_curr_device, x_curr, nDynVert);
			// copy_thrustvector2mat(solver_data_device->b_curr_device, b_curr, nDynVert);
            // GS_global(b_curr, x_curr, 45);
			/*tbb::parallel_invoke(
				[&]() {
					x_curr.block(0, 0, nDynVert, 1) = m_LLT_solver->solve(b_curr.col(0));
				},
				[&]() {
					x_curr.block(0, 1, nDynVert, 1) = m_LLT_solver->solve(b_curr.col(1));
				},
				[&]() {
					x_curr.block(0, 2, nDynVert, 1) = m_LLT_solver->solve(b_curr.col(2));
				}
			);*/
			//copy_mat2thrustvector(x_curr, solver_data_device->x_curr_device, nDynVert);

		}

		//DX = m_D * x_curr.block(0, 0, nDynVert, 3);
		//m_Ue += DX - z;
		op_Ax(*m_D_device, solver_data_device->x_curr_device, DX_device);
		op_a_minus_b(DX_device, solver_data_device->z_buffer, cache_nCDimX3, m_nCDim);
		op_a_plus_b(cache_nCDimX3, m_Ue_device, m_Ue_device, m_nCDim);

		if (enable_frictional_contact) {
			// m_Uc += x_curr.block(0, 0, nDynVert, 3) - x_0.block(0, 0, nDynVert, 3) - p.block(0, 0, nDynVert, 3) * m_dt;
			op_a_minus_b(solver_data_device->x_curr_device, solver_data_device->x_0_device, cache_nDynVertX3, nDynVert);
			op_scale(solver_data_device->p_device, dt_r, cache_nDynVertX3_bp1, nDynVert);
			op_a_minus_b(cache_nDynVertX3, cache_nDynVertX3_bp1, cache_nDynVertX3, nDynVert);
			op_a_plus_b(m_Uc_device, cache_nDynVertX3, m_Uc_device, nDynVert);
		}

		
	}

	// copy x_curr to x_0
	// get v_0
	do_post_process(m_nVert, nDynVert, m_dt, solver_data_device->v_0_device.data().get(), 
		solver_data_device->x_curr_device.data().get(), solver_data_device->x_0_device.data().get());

	// x_0 to m_vertices
	// v_0 to m_velocities
	copy_thrustvector2mat(solver_data_device->x_0_device, m_vertices, m_nVert);
	copy_thrustvector2mat(solver_data_device->v_0_device, m_velocities, m_nVert);

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

void ADMMParallelSolver::Jacobi_global(int iter_cnt) {
	using namespace ADU;

	int nDynVert = static_vert_begin;

	cu_jacobi_global(solver_data_device->sp_mat_device,
		solver_data_device->b_curr_device,
		solver_data_device->x_curr_device,
		solver_data_device->jacobi_buffer_1, solver_data_device->jacobi_buffer_2, nDynVert, iter_cnt);
}

void ADMMParallelSolver::project_feasible_parallel()
{
	 ADMMParallelSolver::_project_feasible_impl();
	//ADMMParallelSolver::_project_feasible_plain(p, contacts, mu, max_GS_iter);
	//ADMMParallelSolver::_project_feasible_plain_2(p, contacts, mu, max_GS_iter);
}

void ADMMParallelSolver::_project_feasible_impl() 
{
	using namespace ADU;
	size_t nContact = bvh->h_cpNum;
	project_impl(bvh, ct_data_device.get(), solver_data_device.get(), gs_max_iter, static_vert_begin, mu, static_vert_begin);
}

void ADMMParallelSolver::_project_feasible_plain(ADU::Matf_X3& p,
		ProximalQuery::ContactInfoList& contacts,
		ADU::Real mu, size_t max_iter)
{
	// p: current velocity
	// notion: r_c lies in contact_info
	using namespace ADU;

	// CT: vinds, r_c, bary, normal
	// admm: delta_u

	size_t nContact = contacts.size();
	for (int ci = 0; ci < nContact; ci++) {
		const auto& ct = contacts[ci];
		for (size_t k = 0; k < 4; k++) {
			size_t j = ct.vinds[k];
			if (j < static_vert_begin) {
				p.row(j) += Gamma_c[ci](k) * ct.r_c;
			}
		}
	}
	


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

					p.row(vi) += delta_pi;
				}
			});
		});


	}
}


void  ADMMParallelSolver::compute_Scc_parallel() {
	compute_Scc_impl();
	return;
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
					Gamma_i[vi].emplace_back(0);
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

	computeContactCountsWithAtomic(bvh->d_collisonPairs, ct_data_device->d_vi_ct_nums, nContact, m_nVert);

	compute_Scc_impl_cu(bvh,
		nContact, this->m_dt_inv, gamma, epsilon,
		ct_data_device.get());

	return;
}


std::vector<std::array<float, 3>>& ADMMParallelSolver::getContactPoints() {
	std::fill(contact_points.begin(), contact_points.end(), std::array<float, 3>{ 0, 0, 0 });
	CUDA_SAFE_CALL(cudaMemcpy(contact_points.data(), ct_data_device->d_point.data().get(), sizeof(float)*3*bvh->h_cpNum, cudaMemcpyDeviceToHost));
	return this->contact_points;
}

std::vector<std::array<float, 3>>& ADMMParallelSolver::getContactNormals() {
	std::fill(contact_normals.begin(), contact_normals.end(), std::array<float, 3>{ 0, 0, 0 });
	CUDA_SAFE_CALL(cudaMemcpy(contact_normals.data(), ct_data_device->d_normal.data().get(), sizeof(float) * 3 * bvh->h_cpNum, cudaMemcpyDeviceToHost));
	return this->contact_normals;
}