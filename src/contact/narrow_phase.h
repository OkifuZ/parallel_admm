#pragma once


#include "mutils/common_types.h"
#include "mesh/mesh_container.h"
#include "contact/contact_parameter.h"
#include "contact/broad_phase.h"
#include "contact/contact_info.h"

#include <tbb/concurrent_vector.h>
#include <memory>
#include <vector>
#include <limits>


class ProximalQuery {
public:

	//using ContactInfoList = std::vector<ContactInfo>;
	using ContactInfoList = tbb::concurrent_vector<ContactInfo>;

	std::shared_ptr<BroadPhase> broad_phase{ nullptr };

	std::shared_ptr<MeshData> mesh{ nullptr };

	ContactInfoList contact_info_list;

	std::vector<std::vector<unsigned int>> candidates_pt;
	std::vector<std::vector<unsigned int>> candidates_ee;

	// dist functor
	ADU::DistEE<ADU::Real> dist_ee;
	ADU::DistPT<ADU::Real> dist_pt;

	const size_t max_collision_num;
	size_t hist_max_collision_num = std::numeric_limits<size_t>::min();

	bool enable_parallel = true;


	ProximalQuery(const std::shared_ptr<MeshData>& mesh, size_t max_collision_num = 4096)
	: max_collision_num(max_collision_num) {
		this->mesh = mesh;
		contact_info_list.reserve(max_collision_num);

		candidates_pt.resize(mesh->surface_vinds.rows());
		candidates_ee.resize(mesh->surface_edges.rows());
		printf("ProximlaQuery point size: %d\n", candidates_pt.size());
		printf("ProximlaQuery edge size: %d\n", candidates_ee.size());
	}

	void proximal_query(const ADU::Matf_X3& pos);
	#ifdef USE_CCD
	void proximal_query_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos);
	#endif

	int unique_contact_size = 0;
	void unique_contact();

	ADU::Matf_X3 contact_points = ADU::Matf_X3(max_collision_num, 3);
	ADU::Matf_X3 S_vecs = ADU::Matf_X3(max_collision_num, 3);
	ADU::Matf_X3 contact_normals = ADU::Matf_X3(max_collision_num, 3);

	void narrow_PT(const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3);
	void narrow_PT_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3);

	void narrow_EE(const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3);
	void narrow_EE_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3);

	ADU::Matf_X3& get_contact_points();

	ADU::Matf_X3& get_contact_normals();

	ADU::Matf_X3 get_SaSb_mean() {
		S_vecs.setZero();

		auto& contacts = contact_info_list;
		for (int ci = 0; ci < contacts.size(); ci++) {
			auto& ct = contacts[ci];
			S_vecs.row(ci) = (ct.Sa + ct.Sb) / 2;
		}
		return S_vecs;

	}

	
private:
	void query_point_triangle(const ADU::Matf_X3& pos);
	void query_point_triangle_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos);

	void query_edge_edge(const ADU::Matf_X3& pos);
	void query_edge_edge_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos);

};