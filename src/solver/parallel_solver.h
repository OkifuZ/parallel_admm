#pragma once
#include "solver/admm_full_solver.h"
#include "constraint/bending_constraint_device.cuh"
#include "constraint/tetrahedral_constraint_device.cuh"


class ContactDataDevice {
public:
	thrust::device_vector<int> d_vi_ct_nums;
	thrust::device_vector<int> d_vi_ct_count;

	thrust::device_vector<float4> d_bary;
	thrust::device_vector<float3> d_normal;
	thrust::device_vector<float3> d_point;
	thrust::device_vector<ADU::Real> d_h_cN;
	//thrust::device_vector<ADU::Real> d_dist;
	thrust::device_vector<int> d_pair_type;

	thrust::device_vector<float4> d_Gamma_c; // 
	thrust::device_vector<float3> d_K_c;
	thrust::device_vector<float3> d_delta_u;

	thrust::device_vector<float3> d_r_c;


	thrust::device_vector<ADU::Real> d_Gamma_i;
	thrust::device_vector<int> d_start_idx;
	thrust::device_vector<int> d_involved_cid;

	thrust::device_vector<ADU::Real> d_contact_W_list;
	thrust::device_vector<ADU::Real> d_contact_W_inv_list;

	ContactDataDevice(int max_Vert, int max_Contact) {
		// size of vert
		d_vi_ct_count.reserve(max_Vert + 10);
		d_vi_ct_nums.reserve(max_Vert + 10);
		d_start_idx.reserve(max_Vert + 10);
		d_contact_W_list.reserve(max_Vert + 10);
		d_contact_W_inv_list.reserve(max_Vert + 10);

		// size of contact
		d_bary.reserve(max_Contact);
		d_normal.reserve(max_Contact);
		d_point.reserve(max_Contact);
		d_h_cN.reserve(max_Contact);
		d_pair_type.reserve(max_Contact);
		d_Gamma_c.reserve(max_Contact);
		d_K_c.reserve(max_Contact);
		d_delta_u.reserve(max_Contact);
		d_r_c.reserve(max_Contact);

		// size of total_contact
		d_involved_cid.reserve(max_Vert * 30);
		d_Gamma_i.reserve(max_Vert * 30);
	}

};

class ADMMParallelSolver : public ADMMSolverFull_RL_damping {

	virtual void step();
	virtual void init();
	virtual void precompute();

	std::unique_ptr<CompactSparseMat> comp_mat;
	std::unique_ptr<CuSolverData> solver_data_device;

	std::unique_ptr<CompactSparseMat> m_D_device;
	std::unique_ptr<CompactSparseMat> m_dt2DTWeTWe_device;
	std::unique_ptr<CompactSparseMat> m_M_device;
	std::unique_ptr<CompactSparseMat> m_dt2Wc_device;


	std::unique_ptr<TriangleConstraintDevice> triangle_constraint_cu;
	int triangle_constraint_start_row{};
	std::unique_ptr<PinConstraintDevice> pin_constraint_cu;
	int pin_constraint_start_row{};
	std::unique_ptr<BendingConstraintDevice> bending_constraint_cu;
	int bending_constraint_start_row{};
	std::unique_ptr<TetrahedralConstraintDevice> tet_constraint_cu;
	int tet_constraint_start_row{};

	thrust::device_vector<ADU::Real> DX_device;
	thrust::device_vector<ADU::Real> m_Ue_device;
	thrust::device_vector<ADU::Real> m_Uc_device;

	thrust::device_vector<ADU::Real> M_x_tilde_device;

	thrust::device_vector<ADU::Real> cache_nDynVertX3;
	thrust::device_vector<ADU::Real> cache_nDynVertX3_bp1;
	thrust::device_vector<ADU::Real> cache_nCDimX3;



	void GS_global(const ADU::Matf_X3& b, ADU::Matf_X3& x_curr, int iter_cnt);
	void Jacobi_global(int iter_cnt);

	ADU::Matf_X3 jacobi_buffer;

	ADU::Matf_X3 x_curr;

	ADU::Matf_X3 p;

	ADU::Matf_X3 b_curr;



	/*void project_feasible(ADU::Matf_X3& p,
		ProximalQuery::ContactInfoList& contacts,
		ADU::Real mu, size_t max_jacobi_iter) override;*/

	void _project_feasible_plain(ADU::Matf_X3& p,
		ProximalQuery::ContactInfoList& contacts,
		ADU::Real mu, size_t max_jacobi_iter) override;

	void project_feasible_parallel();
	void _project_feasible_impl();

	std::vector<tbb::concurrent_vector<ADU::Real>> Gamma_i;
	std::vector<ADU::Vecf_4> Gamma_c_aux; // original gamma_c, being ot divided by ni
	std::vector<tbb::concurrent_vector<int>> involved_cid;
	std::vector<ADU::Vecf_3> delta_u;
	std::vector<int> vi_ct_nums;

	//std::vector<tbb::concurrent_vector<ADU::Real>> Gamma_i;
	//std::vector<ADU::Vecf_4> Gamma_c_aux; // original gamma_c, being ot divided by ni
	//std::vector<tbb::concurrent_vector<int>> involved_cid;
	//std::vector<ADU::Vecf_3> delta_u;
	//thrust::device_vector<int> d_vi_ct_nums;
	//thrust::device_vector<int> d_vi_ct_count;

	//thrust::device_vector<float4> d_bary;
	//thrust::device_vector<float3> d_normal;
	//thrust::device_vector<float3> d_point;
	//thrust::device_vector<ADU::Real> d_h_cN;
	//thrust::device_vector<ADU::Real> d_dist;
	//thrust::device_vector<int> d_pair_type;

	//thrust::device_vector<float4> d_Gamma_c; // 
	//thrust::device_vector<float3> d_K_c;
	//thrust::device_vector<float3> d_delta_u;

	// gamma_i, involved_cid needs special care!!!

	virtual void compute_Scc(bool is_XPBD = false);
	void compute_Scc_parallel();

	/*thrust::device_vector<ADU::Real> d_Gamma_i;
	thrust::device_vector<int> d_start_idx;
	thrust::device_vector<int> d_involved_cid;

	thrust::device_vector<ADU::Real> d_contact_W_list;*/
	std::shared_ptr<ContactDataDevice> ct_data_device{};

	void compute_Scc_impl();


	std::vector<std::array<float, 3>> contact_points;
	std::vector<std::array<float, 3>> contact_normals;

	thrust::device_vector<int> d_is_fixed;
	thrust::device_vector<ADU::Real> d_M;

	virtual void reset(const ADU::Matf_X3& ini_verts, bool need_precompute = false);

public:
	void convert_constraint2device();

	int Global_Jacobi_iter = 20;


	std::vector<std::array<float, 3>>& getContactPoints();
	std::vector<std::array<float, 3>>& getContactNormals();

	


};