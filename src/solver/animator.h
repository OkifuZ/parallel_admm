#pragma once

#include "mesh/mesh_container.h"
#include "mutils/common_types.h"
#include "mutils/cnpy.h"
#include <vector>
#include <memory>
#include <map>
#include <string>





class ScriptAnimator {
public:
	std::shared_ptr<MeshData> mesh;

	enum class SCR_TYPE {
		RIGID,
		POINTS
	};

	struct ScriptedData {
		SCR_TYPE type;

		std::vector<ADU::Vecf_3> pos;
		std::vector<ADU::Quatf> quat;

		std::vector<ADU::Matf_X3> points_data;

		size_t frame_nu;
	};
	
	using ScriptDataPtr = std::shared_ptr<ScriptedData>;
	std::vector<ScriptedData> m_script_data;

	std::map<int, int> m_mesh_id_2_script_idx;
	
	void reset();

	void attach_animate2mesh(int mesh_id, const std::string& npz_path);
	void read_rigid_animate(int mesh_id, const std::string& npz_path);
	void read_point_animate(int mesh_id, const std::string& npz_path);

	int m_curr_frame = -1;
	void animate(int mesh_id, const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt);
	void animate_rigid(int mesh_id, const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt);
	void animate_points(int mesh_id, const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt);
	void animate_all(const ADU::Matf_X3& verts_prev, ADU::Matf_X3& verts, ADU::Matf_X3& velocity, ADU::Real dt);
};