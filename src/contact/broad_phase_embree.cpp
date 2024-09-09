#include "contact/broad_phase.h"
#include "contact/contact_parameter.h"
#include <tbb/parallel_for.h>

#include <unordered_set>


void _BroadPhase_errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	printf("error %d: %s\n", error, str);
}

MeshData* _broad_phase_mesh_ptr{nullptr};

bool closestPointFunc_tri(RTCPointQueryFunctionArguments* args) {
	using namespace ADU;

	//const unsigned int geomID = args->geomID;
	//const unsigned int primID = args->primID;

	BroadPhase_embree::PointProxQueryResult* res = (BroadPhase_embree::PointProxQueryResult*)args->userPtr;

	res->prims.push_back(args->primID);
	res->type.push_back(0);

#ifdef _BVH_BETTER_CULLING
	const auto& verts = _broad_phase_mesh_ptr->verts;
	const auto& tvinds = _broad_phase_mesh_ptr->surface_tris.row(primID);
	const auto& v0 = verts.row(res->vind);
	const auto& v1 = verts.row(tvinds(0));
	const auto& v2 = verts.row(tvinds(1));
	const auto& v3 = verts.row(tvinds(2));

	auto t_center = (v1 + v2 + v3) / 3.0_r;

	auto d = (t_center - v0).norm();
	if (d < ContactParameter::get_broadphase_radius()) {
		args->query->radius = d;
		return true;
	}
#endif // _BVH_BETTER_CULLING

	return false;
}

bool closestPointFunc_edge(RTCPointQueryFunctionArguments* args) {
	using namespace ADU;

	//const unsigned int geomID = args->geomID;
	//const unsigned int primID = args->primID;

	BroadPhase_embree::PointProxQueryResult* res = (BroadPhase_embree::PointProxQueryResult*)args->userPtr;

	res->prims.push_back(args->primID);
	res->type.push_back(1);

#ifdef _BVH_BETTER_CULLING
	const auto& verts = _broad_phase_mesh_ptr->verts;
	const auto& evinds_tar = _broad_phase_mesh_ptr->surface_edges.row(primID);
	const auto& evinds_src = _broad_phase_mesh_ptr->surface_edges.row(res->eind);
	const auto& v0 = verts.row(evinds_tar(0));
	const auto& v1 = verts.row(evinds_tar(1));
	const auto& v2 = verts.row(evinds_src(0));
	const auto& v3 = verts.row(evinds_src(1));

	auto d = ((v0 + v1 - v2 - v3) * 0.5_r).norm();
	if (d < ContactParameter::get_broadphase_radius()) {
		args->query->radius = d;
		return true;
	}
#endif // _BVH_BETTER_CULLING

	return false;
}

BroadPhase_embree::BroadPhase_embree(BroadPhase_embree::BVH_TYPE bvh_type_) {

	RTCDevice device = rtcNewDevice(NULL);
	if (!device) {
		printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));
	}
	rtcSetDeviceErrorFunction(device, _BroadPhase_errorFunction, NULL);

	this->bvh_type = bvh_type;

	scene = rtcNewScene(device);
	//rtcSetSceneFlags(scene, RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
	rtcSetSceneFlags(scene, RTC_SCENE_FLAG_DYNAMIC );
	rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_LOW);

	tri_geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	edge_geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	if (bvh_type == REFIT) {
		rtcSetGeometryBuildQuality(tri_geom, RTC_BUILD_QUALITY_REFIT);
		rtcSetGeometryBuildQuality(edge_geom, RTC_BUILD_QUALITY_REFIT);
	}
	else {
		rtcSetGeometryBuildQuality(tri_geom, RTC_BUILD_QUALITY_LOW);
		rtcSetGeometryBuildQuality(edge_geom, RTC_BUILD_QUALITY_LOW);
	}

	rtcSetGeometryPointQueryFunction(tri_geom, closestPointFunc_tri);
	rtcSetGeometryPointQueryFunction(edge_geom, closestPointFunc_edge);
	rtcCommitGeometry(tri_geom);
	rtcCommitGeometry(edge_geom);

	rtcAttachGeometry(scene, tri_geom);
	rtcAttachGeometry(scene, edge_geom);
	rtcReleaseGeometry(tri_geom);
	rtcReleaseGeometry(edge_geom);

	this->radius = ContactParameter::get_broadphase_radius();

}

void BroadPhase_embree::buildBVH() {
	using namespace ADU;
	if (!mesh) make_exception("buildBVH empty mesh");

	_broad_phase_mesh_ptr = mesh.get();

	verts = mesh->verts.cast<float>();
	tinds = mesh->surface_tris.cast<unsigned int>();
	size_t n_verts = verts.rows();
	size_t n_tris = tinds.rows();

	// TODO: verts should be 16 byte aligned, but seems no matter
	rtcSetSharedGeometryBuffer(tri_geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, verts.data(), 0, 3 * sizeof(float), n_verts);
	rtcSetSharedGeometryBuffer(tri_geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, tinds.data(), 0, 3 * sizeof(unsigned int), n_tris);
	rtcCommitGeometry(tri_geom);

	auto einds = mesh->surface_edges;
	size_t n_edges = einds.rows();
	einds_mock.resize(n_edges * 3); // this has to be static
	for (int ei = 0; ei < n_edges; ei++) {
		const auto& evinds = mesh->surface_edges.row(ei);
		einds_mock[ei * 3 + 0] = evinds(0);
		einds_mock[ei * 3 + 1] = evinds(1);
		einds_mock[ei * 3 + 2] = evinds(1);
	}

	rtcSetSharedGeometryBuffer(edge_geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, verts.data(), 0, 3 * sizeof(float), n_verts);
	rtcSetSharedGeometryBuffer(edge_geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, einds_mock.data(), 0, 3 * sizeof(unsigned int), n_edges);
	rtcCommitGeometry(edge_geom);

	rtcCommitScene(scene);
}

void BroadPhase_embree::updateBVH(const ADU::Matf_X3& new_verts) {
	using namespace ADU;

	if (!mesh) make_exception("updateBVH empty mesh");

	verts = new_verts.cast<float>();
	rtcCommitGeometry(tri_geom);
	rtcCommitGeometry(edge_geom);
	rtcCommitScene(scene);
}

BroadPhase_embree::PointProxQueryResult BroadPhase_embree::query_point(const ADU::Vecf_3& pt) {
	RTCPointQuery query;
	query.x = (float)pt.x();
	query.y = (float)pt.y();
	query.z = (float)pt.z();
	query.radius = this->radius;
	query.time = 0.0f;
	PointProxQueryResult result;
	RTCPointQueryContext context;
	rtcInitPointQueryContext(&context);
	rtcPointQuery(scene, &(query), &context, nullptr, (void*)&result);
	return result;
}

BroadPhase_embree::PointProxQueryResult BroadPhase_embree::query_point(const ADU::Vecf_3& pt, size_t vid) {
	RTCPointQuery query;
	query.x = (float)pt.x();
	query.y = (float)pt.y();
	query.z = (float)pt.z();
	query.radius = this->radius;
	query.time = 0.0f;
	PointProxQueryResult result;
	RTCPointQueryContext context;
	rtcInitPointQueryContext(&context);
	rtcPointQuery(scene, &(query), &context, nullptr, (void*)&result);
	return result;
}

void BroadPhase_embree::query_point_triangle(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) {
	using namespace ADU;

	const auto& surf_vinds = mesh->surface_vinds;
	size_t n_verts = surf_vinds.rows();

	/*candidates.resize(n_verts);*/
	if (candidates.size() != n_verts) {
		ADU::make_exception("BroadPhase::query_point_triangle candidate size error");
	}

	RTCPointQueryContext context;
	rtcInitPointQueryContext(&context);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n_verts), [&](tbb::blocked_range<size_t> r) {
		PointProxQueryResult result;
		RTCPointQuery query;
		for (size_t qi = r.begin(); qi < r.end(); qi++) {
			const auto& v = verts.row(surf_vinds(qi));
			query.x = v.x();
			query.y = v.y();
			query.z = v.z();
			query.radius = this->radius;
			query.time = 0.0f;
			// context move out, correct?
			result.vind = surf_vinds(qi);
			rtcPointQuery(scene, &(query), &context, nullptr, (void*)&result);
			auto& cand = candidates[qi];
			for (size_t ri = 0; ri < result.prims.size(); ri++) {
				if (result.type[ri] == 0) {
					cand.push_back(result.prims[ri]);
				}
			}
			result.prims.clear();
			result.type.clear();
			result.vind = 0;
		}
		});
}

void BroadPhase_embree::query_edge_edge(const ADU::Matf_X3& verts, std::vector<std::vector<unsigned int>>& candidates) {
	using namespace ADU;

	const auto& surf_edges = mesh->surface_edges;
	size_t n_edges = surf_edges.rows();
	const auto& surf_tri_eids = mesh->surf_tri_eids;

	if (candidates.size() != n_edges) {
		ADU::make_exception("BroadPhase::query_edge_edge candidate size error");
	}

	RTCPointQueryContext context;
	rtcInitPointQueryContext(&context);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n_edges), [&](tbb::blocked_range<size_t> r) {
		PointProxQueryResult result;
		RTCPointQuery query;
		for (size_t ei = r.begin(); ei < r.end(); ei++) {
			const auto& evinds = surf_edges.row(ei);
			const auto& v1 = verts.row(evinds(0));
			const auto& v2 = verts.row(evinds(1));
			const auto v = (v1 + v2) * 0.5_r;
			const Real el = (v1 - v2).norm();
			query.x = v.x();
			query.y = v.y();
			query.z = v.z();
			query.radius = el * 0.5 + this->radius * 0.5;
			query.time = 0.0f;
			// context move out, correct?
			result.eind = ei;
			rtcPointQuery(scene, &(query), &context, nullptr, (void*)&result);
			auto& cand = candidates[ei];
			for (size_t ri = 0; ri < result.prims.size(); ri++) {
				if (result.type[ri] == 1) {
					cand.push_back(result.prims[ri]);
				}
			}
			result.prims.clear();
			result.type.clear();
			result.eind = 0;
		}
		});
}