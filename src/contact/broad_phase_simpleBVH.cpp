#include "contact/broad_phase.h"
#include "contact/contact_parameter.h"
#include "mutils/common_types.h"
#include "mutils/exception_handle.h"
#include <type_traits>

#include "tbb/parallel_for.h"


BroadPhase_simpleBVH::BroadPhase_simpleBVH() {
	this->radius = ContactParameter::get_broadphase_radius();
}

void BroadPhase_simpleBVH::build() {
	if (!mesh) {
		ADU::make_exception("BroadPhase_simpleBVH empty mesh");
	}
	if (std::is_same<ADU::Real, float>::value) {
		ADU::make_exception("BroadPhase_simpleBVH only supports double");
	}

	edge_aabbs.resize(mesh->surface_edges.rows());
	face_aabbs.resize(mesh->surface_tris.rows());
}

void BroadPhase_simpleBVH::update(const ADU::Matf_X3& new_verts) {
	clear();

	build_face_boxes(new_verts);
	build_edge_boxes(new_verts);

	face_bvh.init(face_aabbs);
	edge_bvh.init(edge_aabbs);
}

void BroadPhase_simpleBVH::query_point_triangle(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) {
	// after update

	const auto& surf_vinds = mesh->surface_vinds;
	size_t n_verts = surf_vinds.rows();

	if (candidates.size() != n_verts) {
		ADU::make_exception("BroadPhase::query_point_triangle candidate size error");
	}

	const Eigen::Vector3d round{ radius, radius, radius };

	tbb::parallel_for(tbb::blocked_range<size_t>(0, n_verts), 
		[&](tbb::blocked_range<size_t> r) 
		{
			for (size_t qi = r.begin(); qi < r.end(); qi++) {
				const Eigen::Vector3d& v = verts.row(surf_vinds(qi));
				face_bvh.intersect_box(v - round, v + round, candidates[qi]);
			}
		}
	);

}

void BroadPhase_simpleBVH::query_edge_edge(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) {
	// after update
	const auto& surf_edges = mesh->surface_edges;
	size_t n_edges = surf_edges.rows();
	const auto& surf_tri_eids = mesh->surf_tri_eids;

	if (candidates.size() != n_edges) {
		ADU::make_exception("BroadPhase::query_edge_edge candidate size error");
	}

	const Eigen::Vector3d round{ radius / 3, radius / 3, radius / 3 };

	tbb::parallel_for(tbb::blocked_range<size_t>(0, n_edges), 
		[&](tbb::blocked_range<size_t> r) 
		{
			for (size_t ei = r.begin(); ei < r.end(); ei++) {
				const auto& ebox = edge_aabbs[ei];
				edge_bvh.intersect_box(ebox[0] - round, ebox[1] + round, candidates[ei]);
			}
		});
}

inline void get_tri_aabb(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, 
	std::array<Eigen::Vector3d, 2>& aabb) 
{
	aabb[0].x() = std::min({ v0.x(), v1.x(), v2.x() });
	aabb[0].y() = std::min({ v0.y(), v1.y(), v2.y() });
	aabb[0].z() = std::min({ v0.z(), v1.z(), v2.z() });

	aabb[1].x() = std::max({ v0.x(), v1.x(), v2.x() });
	aabb[1].y() = std::max({ v0.y(), v1.y(), v2.y() });
	aabb[1].z() = std::max({ v0.z(), v1.z(), v2.z() });
}

inline void get_edge_aabb(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1,
	std::array<Eigen::Vector3d, 2>& aabb)
{
	aabb[0].x() = std::min({ v0.x(), v1.x()});
	aabb[0].y() = std::min({ v0.y(), v1.y()});
	aabb[0].z() = std::min({ v0.z(), v1.z()});

	aabb[1].x() = std::max({ v0.x(), v1.x()});
	aabb[1].y() = std::max({ v0.y(), v1.y()});
	aabb[1].z() = std::max({ v0.z(), v1.z()});
}

void BroadPhase_simpleBVH::build_face_boxes(const ADU::Matf_X3& new_verts) {
	size_t n_faces = mesh->surface_tris.rows();
	const auto& tris_inds = mesh->surface_tris;
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, n_faces), [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i < r.end(); i++) {
				const auto& tri_inds = tris_inds.row(i);
				get_tri_aabb(new_verts.row(tri_inds(0)), new_verts.row(tri_inds(1)), new_verts.row(tri_inds(2)), face_aabbs[i]);
			}
		},
		tbb::static_partitioner{}
	);
}

void BroadPhase_simpleBVH::build_edge_boxes(const ADU::Matf_X3& new_verts) {
	size_t n_edges = mesh->surface_edges.rows();
	const auto& edges_inds = mesh->surface_edges;
	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, n_edges), [&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i < r.end(); i++) {
				const auto& edge_inds = edges_inds.row(i);
				get_edge_aabb(new_verts.row(edge_inds(0)), new_verts.row(edge_inds(1)), edge_aabbs[i]);
			}
		},
		tbb::static_partitioner{}
	);
}

void BroadPhase_simpleBVH::clear() {
	// undo update, but not build
	edge_bvh.clear();
	face_bvh.clear();
}