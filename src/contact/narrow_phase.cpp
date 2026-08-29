#include "narrow_phase.h"
#include "mutils/distEE.h"
#include "mutils/distPT.h"
#include "mutils/timer.h"
#include "mutils/exception_handle.h"

#ifdef USE_CCD
#include "doubleCCD/doubleccd.hpp"
#endif

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <algorithm>

#include "contact/lbvh/lbvh.cuh"

void ProximalQuery::proximal_query(const ADU::Matf_X3& pos) {
	contact_info_list.clear();

	if (broad_phase) {
		ADU::Timer broadphase_timer("broad_phase");

		const auto& surf_vinds = mesh->surface_vinds;
		for (size_t vi = 0; vi < surf_vinds.rows(); vi++) {
			candidates_pt[vi].clear();
		}
		const auto& surf_einds = mesh->surface_edges;
		for (size_t ei = 0; ei < surf_einds.rows(); ei++) {
			candidates_ee[ei].clear();
		} 
		{
			ADU::Timer updateBVH_timer("updateBVH");
			//broad_phase->updateBVH(pos);
			broad_phase->update(pos);
		}
		{
			ADU::Timer query_point_triangle_timer("query_point_triangle");
			broad_phase->query_point_triangle(pos, candidates_pt);
		}
		{
			ADU::Timer query_edge_edge_timer("query_edge_edge");
			broad_phase->query_edge_edge(pos, candidates_ee);
		}
	}

	ADU::Timer narrowphase_timer("narrow_phase");
	{
		ADU::Timer query_point_triangle_timer("query_point_triangle");
		query_point_triangle(pos);
	}
	{
		ADU::Timer query_edge_edge_timer("query_edge_edge");
		query_edge_edge(pos);
	}

	if (contact_info_list.size() > max_collision_num)
		ADU::make_exception("proximal_query() exceeds max_collision_num !");
	else {
		hist_max_collision_num = std::max(hist_max_collision_num, contact_info_list.size());
	}

	// TODO find island
}

#ifdef USE_CCD
void ProximalQuery::proximal_query_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos) {
	contact_info_list.clear();

	if (broad_phase) {
		ADU::Timer broadphase_timer("broad_phase");

		const auto& surf_vinds = mesh->surface_vinds;
		for (size_t vi = 0; vi < surf_vinds.rows(); vi++) {
			candidates_pt[vi].clear();
		}
		const auto& surf_einds = mesh->surface_edges;
		for (size_t ei = 0; ei < surf_einds.rows(); ei++) {
			candidates_ee[ei].clear();
		}
		{
			ADU::Timer updateBVH_timer("updateBVH");
			//broad_phase->updateBVH(pos);
			broad_phase->update(pos);
		}
		{
			ADU::Timer query_point_triangle_timer("query_point_triangle");
			broad_phase->query_point_triangle(pos, candidates_pt);
		}
		{
			ADU::Timer query_edge_edge_timer("query_edge_edge");
			broad_phase->query_edge_edge(pos, candidates_ee);
		}
	}

	ADU::Timer narrowphase_timer("narrow_phase_with_CCD");
	{
		ADU::Timer query_point_triangle_timer("query_point_triangle_with_CCD");
		//query_point_triangle(pos);
		query_point_triangle_with_CCD(pos_t0, pos);
	}
	{
		ADU::Timer query_edge_edge_timer("query_edge_edge_with_CCD");
		query_edge_edge_with_CCD(pos_t0, pos);
	}

	// TODO: make this compatible with unique()
	if (contact_info_list.size() > max_collision_num)
		ADU::make_exception("proximal_query() exceeds max_collision_num !");
	else {
		hist_max_collision_num = std::max(hist_max_collision_num, contact_info_list.size());
	}

	// TODO find island

}
#endif

void ProximalQuery::unique_contact() {
	using namespace ADU;

	const ContactCompare cmp;
	tbb::parallel_sort(this->contact_info_list.begin(), this->contact_info_list.end(), cmp);
	const auto& unique_end = std::unique(this->contact_info_list.begin(), this->contact_info_list.end());
	unique_contact_size = unique_end - this->contact_info_list.begin();
	if (unique_contact_size > this->contact_info_list.size()) ADU::make_exception("unique_contact unique_size larger than before");
	if (unique_contact_size < this->contact_info_list.size()) {
		//printf("reduced %d\n", this->contact_info_list.size() - unique_contact_size);
		this->contact_info_list.resize(unique_contact_size);
	}
	/*if (this->contact_info_list.size() != unique_contact_size) {
		printf("%d %d\n", this->contact_info_list.size(), unique_contact_size);
	}*/
}

#ifdef USE_CCD
void ProximalQuery::query_point_triangle_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos) {
	using namespace ADU;
	const Mati_X3& faces = mesh->surface_tris;
	const auto& surf_vinds = mesh->surface_vinds;
	size_t nVert = surf_vinds.rows();
	size_t nFace = faces.rows();
	const auto& candidates_pt = this->candidates_pt;
	const auto& is_static = mesh->is_static;

	if (enable_parallel) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nVert), [&](tbb::blocked_range<size_t> r) {
			for (size_t idx = r.begin(); idx < r.end(); idx++) {
				size_t vi = surf_vinds(idx);
				if (broad_phase) {
					for (int fi : candidates_pt[idx]) {
						const Veci_3& f = faces.row(fi);
						if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
						if (is_static(vi) && is_static(f(0))) continue;
						narrow_PT_with_CCD(pos_t0, pos, vi, f(0), f(1), f(2));
					}
				}
				else {
					for (size_t fi = 0; fi < nFace; fi++) {
						const Veci_3& f = faces.row(fi);
						if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
						if (is_static(vi) && is_static(f(0))) continue;
						narrow_PT_with_CCD(pos_t0, pos, vi, f(0), f(1), f(2));
					}
				}
			}
			});
	}
	else {
		for (size_t idx = 0; idx < nVert; idx++) {
			size_t vi = surf_vinds(idx);
			if (broad_phase) {
				for (int fi : candidates_pt[idx]) {
					const Veci_3& f = faces.row(fi);
					if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
					if (is_static(vi) && is_static(f(0))) continue;
					narrow_PT_with_CCD(pos_t0, pos, vi, f(0), f(1), f(2));
				}
			}
			else {
				for (size_t fi = 0; fi < nFace; fi++) {
					const Veci_3& f = faces.row(fi);
					if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
					if (is_static(vi) && is_static(f(0))) continue;
					narrow_PT_with_CCD(pos_t0, pos, vi, f(0), f(1), f(2));
				}
			}
		}
	}
}
#endif

void ProximalQuery::query_point_triangle(const ADU::Matf_X3& pos) {
	using namespace ADU;
	const Mati_X3& faces = mesh->surface_tris;
	const auto& surf_vinds = mesh->surface_vinds;
	size_t nVert = surf_vinds.rows();
	size_t nFace = faces.rows();
	const auto& candidates_pt = this->candidates_pt;
	const auto& is_static = mesh->is_static;

	if (enable_parallel) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nVert), [&](tbb::blocked_range<size_t> r) {
			for (size_t idx = r.begin(); idx < r.end(); idx++) {
				size_t vi = surf_vinds(idx);
				if (broad_phase) {
					for (int fi : candidates_pt[idx]) {
						const Veci_3& f = faces.row(fi);
						if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
						if (is_static(vi) && is_static(f(0))) continue;
						narrow_PT(pos, vi, f(0), f(1), f(2));
					}
				}
				else {
					for (size_t fi = 0; fi < nFace; fi++) {
						const Veci_3& f = faces.row(fi);
						if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
						if (is_static(vi) && is_static(f(0))) continue;
						narrow_PT(pos, vi, f(0), f(1), f(2));
					}
				}
			}
			});
	}
	else {
		for (size_t idx = 0; idx < nVert; idx++) {
			size_t vi = surf_vinds(idx);
			if (broad_phase) {
				for (int fi : candidates_pt[idx]) {
					const Veci_3& f = faces.row(fi);
					if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
					if (is_static(vi) && is_static(f(0))) continue;
					narrow_PT(pos, vi, f(0), f(1), f(2));
				}
			}
			else {
				for (size_t fi = 0; fi < nFace; fi++) {
					const Veci_3& f = faces.row(fi);
					if (vi == f(0) || vi == f(1) || vi == f(2)) continue;
					if (is_static(vi) && is_static(f(0))) continue;
					narrow_PT(pos, vi, f(0), f(1), f(2));
				}
			}
		}
	}
}

#ifdef USE_CCD
void ProximalQuery::query_edge_edge_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos) {
	using namespace ADU;
	const Mati_X2& edges = mesh->surface_edges;
	size_t nEdge = edges.rows();
	const auto& is_static = mesh->is_static;

	if (enable_parallel) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nEdge), [&](tbb::blocked_range<size_t> r) {
			for (size_t ei = r.begin(); ei < r.end(); ei++) {
				if (broad_phase) {
					for (int ej : candidates_ee[ei]) {
						if (ei >= ej) continue;
						const Veci_2& e0 = edges.row(ei);
						const Veci_2& e1 = edges.row(ej);
						if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
							continue;
						if (is_static(e0(0)) && is_static(e1(0))) continue;

						narrow_EE_with_CCD(pos_t0, pos, e0(0), e0(1), e1(0), e1(1));
					}
				}
				else {
					for (size_t ej = ei + 1; ej < nEdge; ej++) {
						const Veci_2& e0 = edges.row(ei);
						const Veci_2& e1 = edges.row(ej);
						if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
							continue;
						if (is_static(e0(0)) && is_static(e1(0))) continue;

						narrow_EE_with_CCD(pos_t0, pos, e0(0), e0(1), e1(0), e1(1));
					}
				}
			}
			});
	}
	else {
		for (size_t ei = 0; ei < nEdge; ei++) {
			if (broad_phase) {
				for (int ej : candidates_ee[ei]) {
					if (ei >= ej) continue;
					const Veci_2& e0 = edges.row(ei);
					const Veci_2& e1 = edges.row(ej);
					if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
						continue;
					if (is_static(e0(0)) && is_static(e1(0))) continue;

					narrow_EE_with_CCD(pos_t0, pos, e0(0), e0(1), e1(0), e1(1));
				}
			}
			else {
				for (size_t ej = ei + 1; ej < nEdge; ej++) {
					const Veci_2& e0 = edges.row(ei);
					const Veci_2& e1 = edges.row(ej);
					if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
						continue;
					if (is_static(e0(0)) && is_static(e1(0))) continue;

					narrow_EE_with_CCD(pos_t0, pos, e0(0), e0(1), e1(0), e1(1));
				}
			}
		}
	}
}
#endif

void ProximalQuery::query_edge_edge(const ADU::Matf_X3& pos) {
	using namespace ADU;
	const Mati_X2& edges = mesh->surface_edges;
	size_t nEdge = edges.rows();
	const auto& is_static = mesh->is_static;

	if (enable_parallel) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nEdge), [&](tbb::blocked_range<size_t> r) {
		for (size_t ei = r.begin(); ei < r.end(); ei++) {
			if (broad_phase) {
				for (int ej : candidates_ee[ei]) {
					if (ei >= ej) continue;
					const Veci_2& e0 = edges.row(ei);
					const Veci_2& e1 = edges.row(ej);
					if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
						continue;
					if (is_static(e0(0)) && is_static(e1(0))) continue;

					narrow_EE(pos, e0(0), e0(1), e1(0), e1(1));
				}
			}
			else {
				for (size_t ej = ei + 1; ej < nEdge; ej++) {
					const Veci_2& e0 = edges.row(ei);
					const Veci_2& e1 = edges.row(ej);
					if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
						continue;
					if (is_static(e0(0)) && is_static(e1(0))) continue;

					narrow_EE(pos, e0(0), e0(1), e1(0), e1(1));
				}
			}
		}
		});
	}
	else {
		for (size_t ei = 0; ei < nEdge; ei++) {
			if (broad_phase) {
				for (int ej : candidates_ee[ei]) {
					if (ei >= ej) continue;
					const Veci_2& e0 = edges.row(ei);
					const Veci_2& e1 = edges.row(ej);
					if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
						continue;
					if (is_static(e0(0)) && is_static(e1(0))) continue;

					narrow_EE(pos, e0(0), e0(1), e1(0), e1(1));
				}
			}
			else {
				for (size_t ej = ei + 1; ej < nEdge; ej++) {
					const Veci_2& e0 = edges.row(ei);
					const Veci_2& e1 = edges.row(ej);
					if (e0(0) == e1(0) || e0(0) == e1(1) || e0(1) == e1(0) || e0(1) == e1(1))
						continue;
					if (is_static(e0(0)) && is_static(e1(0))) continue;

					narrow_EE(pos, e0(0), e0(1), e1(0), e1(1));
				}
			}
		}
	}
}

void ProximalQuery::narrow_PT(const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3) {
	using namespace ADU;
	// point
	const Vecf_3& v0 = pos.row(vi0);
	// triangle
	const Vecf_3& v1 = pos.row(vi1);
	const Vecf_3& v2 = pos.row(vi2);
	const Vecf_3& v3 = pos.row(vi3);

	const auto& pt_result = dist_pt(v0, v1, v2, v3);
	if (pt_result.distance <= ContactParameter::get_thickness()) {
		/*if ((pt_result.barycentric[0] == 0 && pt_result.barycentric[1] == 0) ||
			(pt_result.barycentric[1] == 0 && pt_result.barycentric[2] == 0) ||
			(pt_result.barycentric[0] == 0 && pt_result.barycentric[2] == 0)) 
		{
			return;
		}*/
		// TODO: is it right? 
		Vecf_3 normal = (v2 - v1).cross(v3 - v1).normalized();
		contact_info_list.emplace_back(pt_result, vi0, vi1, vi2, vi3,
			normal,
			0.5_r * (pt_result.closest[0] + pt_result.closest[1]), false);
	}
}

#ifdef USE_CCD
void ProximalQuery::narrow_PT_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3) {
	using namespace ADU;
	// point
	const Vecf_3& v0_t0 = pos_t0.row(vi0);
	const Vecf_3& v0 = pos.row(vi0);
	// triangle
	const Vecf_3& v1_t0 = pos_t0.row(vi1);
	const Vecf_3& v2_t0 = pos_t0.row(vi2);
	const Vecf_3& v3_t0 = pos_t0.row(vi3);
	const Vecf_3& v1 = pos.row(vi1);
	const Vecf_3& v2 = pos.row(vi2);
	const Vecf_3& v3 = pos.row(vi3);

	bool hit = doubleccd::vertexFaceCCD(
		v0_t0, v1_t0, v2_t0, v3_t0, v0, v1, v2, v3);
	//hit = false;
	if (hit) { // move too fast, inverted
		//printf("hit pt\n");
		const auto& pt_result = dist_pt(v0, v1, v2, v3);
		//if (pt_result.distance >= ContactParameter::get_thickness()) printf("hit_PT\n");
		//printf("hit_PT\n");
		Vecf_3 normal = (v2 - v1).cross(v3 - v1).normalized();
		//Vecf_3 normal = (pt_result.closest[0] - pt_result.closest[1]).normalized();
		contact_info_list.emplace_back(pt_result, vi0, vi1, vi2, vi3,
			normal,
			0.5_r * (pt_result.closest[0] + pt_result.closest[1]), true);
	}
	else { // thickness
		const auto& pt_result = dist_pt(v0, v1, v2, v3);
		//const auto& pt_result = dist_pt(v0, v1, v2, v3);
		if (pt_result.distance <= ContactParameter::get_thickness()) {
			Vecf_3 normal = (v2 - v1).cross(v3 - v1).normalized();
			contact_info_list.emplace_back(pt_result, vi0, vi1, vi2, vi3,
				normal,
				0.5_r * (pt_result.closest[0] + pt_result.closest[1]), false);
		}
		return;
	}

}
#endif

void ProximalQuery::narrow_EE(const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3) {
	using namespace ADU;
	// edge 1
	const Vecf_3& v0 = pos.row(vi0);
	const Vecf_3& v1 = pos.row(vi1);
	// edge 2
	const Vecf_3& v2 = pos.row(vi2);
	const Vecf_3& v3 = pos.row(vi3);

	const auto ee_result = dist_ee(v0, v1, v2, v3);
	if (ee_result.distance < ContactParameter::get_thickness()) {
		if ((ee_result.parameter[0] == 0.0_r || ee_result.parameter[0] == 1.0_r) &&
			(ee_result.parameter[1] == 0.0_r || ee_result.parameter[1] == 1.0_r)) 
		/*if ((ee_result.parameter[0] == 0.0_r || ee_result.parameter[0] == 1.0_r) ||
			(ee_result.parameter[1] == 0.0_r || ee_result.parameter[1] == 1.0_r))*/
		{
			return;
		}
		// TODO: is normal direction right? 
		Vecf_3 normal = (v1 - v0).cross(v3 - v2).normalized();
		contact_info_list.emplace_back(ee_result, vi0, vi1, vi2, vi3,
			normal,
			0.5_r * (ee_result.closest[0] + ee_result.closest[1]), false);
	}
}

#ifdef USE_CCD
void ProximalQuery::narrow_EE_with_CCD(const ADU::Matf_X3& pos_t0, const ADU::Matf_X3& pos, int vi0, int vi1, int vi2, int vi3) {
	using namespace ADU;
	// edge 1
	const Vecf_3& v0_t0 = pos_t0.row(vi0);
	const Vecf_3& v1_t0 = pos_t0.row(vi1);
	const Vecf_3& v0 = pos.row(vi0);
	const Vecf_3& v1 = pos.row(vi1);
	// edge 2
	const Vecf_3& v2_t0 = pos_t0.row(vi2);
	const Vecf_3& v3_t0 = pos_t0.row(vi3);
	const Vecf_3& v2 = pos.row(vi2);
	const Vecf_3& v3 = pos.row(vi3);

	bool hit = doubleccd::edgeEdgeCCD(
		v0_t0, v1_t0, v2_t0, v3_t0, v0, v1, v2, v3);

	//hit = false;
	if (hit) { // move too fast, inverted
		//printf("hit ee\n");
		const auto ee_result = dist_ee(v0, v1, v2, v3);
		//if (ee_result.distance >= ContactParameter::get_thickness()) printf("hit_EE\n");
		//printf("hit_EE\n");
		Vecf_3 normal = (v1 - v0).cross(v3 - v2).normalized();
		//Vecf_3 normal = (ee_result.closest[0] - ee_result.closest[1]).normalized();
		contact_info_list.emplace_back(ee_result, vi0, vi1, vi2, vi3,
			normal,
			0.5_r * (ee_result.closest[0] + ee_result.closest[1]), true);
	}
	else { // thickness
		const auto ee_result = dist_ee(v0, v1, v2, v3);
		if (ee_result.distance < ContactParameter::get_thickness()) {
			if ((ee_result.parameter[0] == 0.0_r || ee_result.parameter[0] == 1.0_r) &&
				(ee_result.parameter[1] == 0.0_r || ee_result.parameter[1] == 1.0_r))
				/*if ((ee_result.parameter[0] == 0.0_r || ee_result.parameter[0] == 1.0_r) ||
					(ee_result.parameter[1] == 0.0_r || ee_result.parameter[1] == 1.0_r))*/
			{
				return;
			}
			// TODO: is normal direction right? 
			Vecf_3 normal = (v1 - v0).cross(v3 - v2).normalized();
			contact_info_list.emplace_back(ee_result, vi0, vi1, vi2, vi3,
				normal,
				0.5_r * (ee_result.closest[0] + ee_result.closest[1]), false);
		}
		return;
	}
}
#endif

ADU::Matf_X3& ProximalQuery::get_contact_points() {
	contact_points.setZero();
	for (int ci = 0; ci < contact_info_list.size(); ci++) {
		const auto& ct = contact_info_list[ci];
		contact_points.row(ci) = ct.point;
	}
	return contact_points;
}

ADU::Matf_X3& ProximalQuery::get_contact_normals() {
	contact_normals.setZero();
	for (int ci = 0; ci < contact_info_list.size(); ci++) {
		const auto& ct = contact_info_list[ci];
		contact_normals.row(ci) = ct.normal;
	}
	return contact_normals;
}


ADU::Matf_X3& ProximalQuery::get_contact_edges_points() {
	/*contact_edges_points.setZero();
	int idx = 0;
	for (int ci = 0; ci < contact_info_list.size(); ci++) {
		const auto& ct = contact_info_list[ci];
		if (ct.pair_type == ContactInfo::EE) {

		}
	}
	return contact_points;*/

	return contact_points;

}


