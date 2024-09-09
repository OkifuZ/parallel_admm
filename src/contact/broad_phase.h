#pragma once

#include <mesh/mesh_container.h>
#include <mutils/common_types.h>
#include <mutils/exception_handle.h>
#include "contact/simpleBVH/BVH.hpp"

#include <embree4/rtcore.h>
#include <tbb/concurrent_vector.h>
#include <memory>

constexpr size_t _BroadPhaseResultReservedSize = 128;

//#define _BVH_BETTER_CULLING

class BroadPhase {
public:
	std::shared_ptr<MeshData> mesh{ nullptr };

	ADU::Real radius{ static_cast<ADU::Real>(0.5) };

	virtual void build() {};
	virtual void update(const ADU::Matf_X3& new_verts) {};

	virtual void query_point_triangle(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) {};
	virtual void query_edge_edge(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) {};
};

class BroadPhase_embree : public BroadPhase {
public:
	
	enum BVH_TYPE {
		REBUILD,
		REFIT
	};

	BVH_TYPE bvh_type{ REBUILD };

	BroadPhase_embree(BVH_TYPE bvh_type_);

	virtual void build() { buildBVH(); }
	virtual void update(const ADU::Matf_X3& new_verts) { updateBVH(new_verts); }

	virtual void query_point_triangle(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) ;
	virtual void query_edge_edge(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) ;

	struct PointProxQueryResult
	{
		PointProxQueryResult() {
			prims.reserve(_BroadPhaseResultReservedSize);
			type.reserve(_BroadPhaseResultReservedSize);
		}

		std::vector<size_t> prims;
		std::vector<unsigned int> type; // 0: tri, 1: edge

		size_t vind = 0;
		size_t eind = 0;
	};

private:

	void buildBVH();

	void updateBVH(const ADU::Matf_X3& new_verts);
	
	PointProxQueryResult query_point(const ADU::Vecf_3& pt);
	PointProxQueryResult query_point(const ADU::Vecf_3& pt, size_t vid);

	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> verts;
	Eigen::Matrix<unsigned int, Eigen::Dynamic, 3, Eigen::RowMajor> tinds;
	RTCDevice device{};
	RTCScene scene{};
	RTCGeometry tri_geom{};
	RTCGeometry edge_geom{};

	std::vector<unsigned int> einds_mock;
	//std::vector<PointProxQueryResult> pt_cache;
};


class BroadPhase_simpleBVH : public BroadPhase {
public:
	BroadPhase_simpleBVH();

	std::vector<std::array<Eigen::Vector3d, 2>> edge_aabbs;
	std::vector<std::array<Eigen::Vector3d, 2>> face_aabbs;

	SimpleBVH::BVH edge_bvh;
	SimpleBVH::BVH face_bvh;

	virtual void build();
	virtual void update(const ADU::Matf_X3& new_verts);

	virtual void query_point_triangle(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates);
	virtual void query_edge_edge(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates);

private:
	void build_edge_boxes(const ADU::Matf_X3& new_verts);

	void build_face_boxes(const ADU::Matf_X3& new_verts);

	void clear();
};