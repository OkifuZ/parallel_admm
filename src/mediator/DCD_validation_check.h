#pragma once

#include "mesh/mesh_container.h"
#include "contact/narrow_phase.h"
#include "mutils/common_types.h"


static bool thickness_validation_check(const MeshData& mesh) {
	using namespace ADU;

	Real thickness = ContactParameter::get_thickness();

	const auto& edges = mesh.surface_edges;

	bool pass = true;
	Real min_edge_len = 2023.0_r * 2023.0_r;

	for (int ei = 0; ei < edges.rows(); ei++) {
		const auto& edge = edges.row(ei);
		Vecf_3 ev = mesh.verts.row(edge(0)) - mesh.verts.row(edge(1));
		Real edge_len = ev.norm();
		if (mesh.is_static(edge(0))) continue;
		if (edge_len * 1.0_r < thickness) {
			pass = false;
			min_edge_len = std::min(min_edge_len, edge_len);
		}
	}
	if (!pass) {
		printf("thickness_validation_check: small edge found, length=%f, thickness = %f\n", min_edge_len, thickness);
		return false;
	}
	return true;
}