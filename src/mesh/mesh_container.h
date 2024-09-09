#pragma once

#include "mutils/common_types.h"
#include "mutils/exception_handle.h"
#include "mutils/mesh_io.h"
#include "mutils/aux_color.h"

#include "igl/readOBJ.h"
#include "igl/boundary_facets.h"
#include "igl/adjacency_list.h"
#include "igl/doublearea.h"
#include "igl/unique.h"
#include "igl/edges.h"
#include "igl/volume.h"

#include <string>
#include <vector>
#include <unordered_set>
#include <set>
#include <memory>
#include <filesystem>
#include <unordered_map>



class MeshData {
	// this class is only for initial stage usage, i.e. initial geomotry for simulated mesh
	// this is not the dynamics mesh
	// additional copy from solver will make this class a dynamic mesh, which is easy to implement
	// eg: 
	// this->verts = solver->m_vertices;

public:
	enum MeshType {
		TRI,
		TET,
		STATIC
	};

	class MeshObject {
	public:
		int color_id = 0;

		MeshType type{};

		size_t start_vertIdx{};
		size_t end_vertIdx{};

		size_t start_faceIdx{};
		size_t end_faceIdx{};

		size_t start_tetIdx{};
		size_t end_tetIdx{};

		size_t start_surftriIdx{};
		size_t end_surftriIdx{};

		bool is_static{ false };

		ADU::Quatf m_rotate;
		ADU::Vecf_3 m_translate;
		ADU::Real m_scale;
		ADU::Real m_scale_x;
		ADU::Real m_scale_y;
		ADU::Real m_scale_z;

		ADU::Vecf_3 min_aabb;
		ADU::Vecf_3 max_aabb;


		MeshObject() {}
	};

	void compuite_AABB() {
		using namespace ADU;
		for (auto& mc : mesh_list) {
			int st_idx = mc->start_vertIdx;
			int ed_idx = mc->end_vertIdx;
			Vecf_3 min_vec{ 1e7, 1e7, 1e7 };
			Vecf_3 max_vec{ -1e7, -1e7, -1e7 };
			for (int i = st_idx; i < ed_idx; i++) {
				const Vecf_3& pos = verts.row(i);
				min_vec.x() = std::min(min_vec.x(), pos.x());
				min_vec.y() = std::min(min_vec.y(), pos.y());
				min_vec.z() = std::min(min_vec.z(), pos.z());
				max_vec.x() = std::max(max_vec.x(), pos.x());
				max_vec.y() = std::max(max_vec.y(), pos.y());
				max_vec.z() = std::max(max_vec.z(), pos.z());
			}
			mc->min_aabb = min_vec;
			mc->max_aabb = max_vec;
		}
	}

	void print_AABB() {
		using namespace ADU;
		int id = 0;
		for (auto& mc : mesh_list) {
			std::cout << "AABB of " << id++ << ": (" << mc->min_aabb.transpose() << ", " << mc->max_aabb.transpose() << ")\n";
		}
	}

	ADU::Mati_X3 getFaceInds(int mesh_id) {
		using namespace ADU;
		if (mesh_id >= mesh_list.size()) ADU::make_exception("getFaceInds out of inidices");
		const auto& mesh_obj = mesh_list[mesh_id];
		Mati_X3 mesh_faces(mesh_obj->end_faceIdx - mesh_obj->start_faceIdx, 3);
		for (int fi = mesh_obj->start_faceIdx; fi < mesh_obj->end_faceIdx; fi++) {
			const auto& finds = this->faces.row(fi);
			mesh_faces.row(fi - mesh_obj->start_faceIdx) = Veci_3{
				finds.x() - (int)mesh_obj->start_vertIdx,
				finds.y() - (int)mesh_obj->start_vertIdx,
				finds.z() - (int)mesh_obj->start_vertIdx
			};
		}
		return mesh_faces;
	}

	ADU::Vecf_3 getMassCenter(int mesh_id, ADU::Vecf_X& mass, ADU::Matf_X3& pos) {
		using namespace ADU;
		if (mesh_id >= mesh_list.size()) ADU::make_exception("getMassCenter out of inidices");
		const auto& mesh_obj = mesh_list[mesh_id];
		Vecf_3 center = Vecf_3::Zero();
		Real total_mass = 0.0_r;
		for (int vi = mesh_obj->start_vertIdx; vi < mesh_obj->end_vertIdx; vi++) {
			total_mass += mass(vi);
			center += mass(vi) * pos.row(vi);
		}
		return center / total_mass;
	}

	ADU::Vecf_3 getMassWeightedVelocity(int mesh_id, ADU::Vecf_X& mass, ADU::Matf_X3& vel) {
		using namespace ADU;
		if (mesh_id >= mesh_list.size()) ADU::make_exception("getMassCenter out of inidices");
		const auto& mesh_obj = mesh_list[mesh_id];
		Vecf_3 vel_center = Vecf_3::Zero();
		Real total_mass = 0.0_r;
		for (int vi = mesh_obj->start_vertIdx; vi < mesh_obj->end_vertIdx; vi++) {
			total_mass += mass(vi);
			vel_center += mass(vi) * vel.row(vi);
		}
		return vel_center / total_mass;
	}


	int static_mesh_id_begin{};
	int static_vert_begin{};

	std::vector<std::shared_ptr<MeshObject>> mesh_list;

	ADU::Veci_X is_static;

	ADU::Matf_X3 verts;
	ADU::Vecf_X verts_density;
	ADU::Matf_X3 v_colors;
	ADU::Matf_X3 f_colors;

	ADU::Mati_X3 faces;
	ADU::Vecf_X area;
	std::vector< std::vector< int > > adjacent_lists; // triangle mesh only, verts belongs to tet will be empty
	ADU::Mati_XX boundary_edges;
	ADU::Mati_X2 tri_edges;
	ADU::Veci_X boundary_vinds_edges;
	
	ADU::Mati_X4 tets;
	ADU::Vecf_X volume;
	ADU::Mati_XX boundary_tris;
	ADU::Mati_X2 tet_edges;
	ADU::Mati_XX boundary_tet_edges;
	ADU::Veci_X boundary_vinds_tris;

	ADU::Mati_X2 surface_edges; // tri_edges + tet_boundary_edges
	ADU::Mati_X3 surface_tris; // tri + tet_boundary
	ADU::Veci_X surface_vinds; // tri + tet_boundary

	
private:
	struct PairHash {
		template <class T1, class T2>
		std::size_t operator () (const std::pair<T1, T2>& p) const {
			auto h1 = std::hash<T1>{}(p.first);
			auto h2 = std::hash<T2>{}(p.second);

			return h1 ^ h2;
		}
	};

	struct PairEqual {
		template <class T1, class T2>
		bool operator() (const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) const {
			return (p1.first == p2.first && p1.second == p2.second) ||
				(p1.first == p2.second && p1.second == p2.first);
		}
	};

public:
	std::unordered_map<std::pair<int, int>, int, PairHash, PairEqual> surf_edge_id_map;
	ADU::Mati_X3 surf_tri_eids;


	MeshData() {}

	size_t add_mesh(const std::string& file_path, MeshType type, ADU::Real density=1, int color_id = 0) {
		std::shared_ptr<MeshObject> mesh = std::make_shared<MeshObject>();
		mesh->type = type;

		mesh->color_id = color_id;

		if (!std::filesystem::exists(file_path)) throw std::runtime_error("Error add_mesh: invalid filepath");

		if (type == TRI) process_triangle_mesh(file_path, *mesh, density);
		else if (type == TET) process_tetrahedral_mesh(file_path, *mesh, density);
		else if (type == STATIC) process_triangle_mesh(file_path, *mesh, density);

		size_t mesh_id = mesh_list.size();
		mesh_list.push_back(mesh);

		return mesh_id;
	}

	void clear_color() {
		using namespace ADU;
		/*for (int vi = 0; vi < v_colors.rows(); vi++) {
			v_colors.row(vi) = CoolColor::color(0);
		}
		for (int fi = 0; fi < f_colors.rows(); fi++) {
			f_colors.row(fi) = CoolColor::color(0);
		}*/
	}


	void precompute_geometry_info() {
		using namespace ADU;

		if (faces.rows() > 0) {
			igl::adjacency_list(faces, adjacent_lists);

			igl::boundary_facets(faces, boundary_edges);
			igl::unique(boundary_edges, boundary_vinds_edges);
			for (int vi = 0; vi < boundary_vinds_edges.rows(); vi++) {
				m_boundary_vinds_set.insert(boundary_vinds_edges(vi));
			}
			
			igl::doublearea(verts, faces, area);
			area *= 0.5_r;

			igl::edges(faces, tri_edges);
		}

		if (tets.rows() > 0) {
			igl::boundary_facets(tets, boundary_tris);
			igl::unique(boundary_tris, boundary_vinds_tris);
			for (int vi = 0; vi < boundary_vinds_tris.rows(); vi++) {
				m_boundary_vinds_set.insert(boundary_vinds_tris(vi));
			}

			igl::volume(verts, tets, volume);

			igl::edges(tets, tet_edges);
			igl::edges(boundary_tris, boundary_tet_edges);
		}

		/*surface_tris.resize(faces.rows() + boundary_tris.rows(), 3);
		if (boundary_tris.rows() > 0 && faces.rows() > 0) {
			surface_tris << faces, boundary_tris;
		}
		else if (faces.rows() > 0) {
			surface_tris = faces;
		}
		else if (boundary_tris.rows() > 0) {
			surface_tris = boundary_tris;
		}
		else {
			ADU::make_exception("Eprecompute_geometry_info: empty surface face");
		}*/

		std::set<int> surf_vinds_set;
		for (int ti = 0; ti < surface_tris.rows(); ti++) {
			const auto& vinds = surface_tris.row(ti);
			surf_vinds_set.insert(vinds(0));
			surf_vinds_set.insert(vinds(1));
			surf_vinds_set.insert(vinds(2));
		}
		surface_vinds.resize(surf_vinds_set.size());
		int idx = 0;
		for (auto it = surf_vinds_set.begin(); it != surf_vinds_set.end(); it++) {
			surface_vinds(idx++) = *it;
		}


		surface_edges.resize(tri_edges.rows() + boundary_tet_edges.rows(), 2);
		if (boundary_tet_edges.rows() > 0 && tri_edges.rows() > 0) {
			surface_edges << tri_edges, boundary_tet_edges;
		}
		else if (tri_edges.rows() > 0) {
			surface_edges = tri_edges;
		}
		else if (boundary_tet_edges.rows() > 0) {
			surface_edges = boundary_tet_edges;
		}
		else {
			ADU::make_exception("Eprecompute_geometry_info: empty surface edge");
		}

		v_colors.resizeLike(verts);
		for (int vi = 0; vi < v_colors.rows(); vi++) {
			v_colors.row(vi) = CoolColor::color(0);
		}

		f_colors.resizeLike(surface_tris);
		for (int vi = 0; vi < f_colors.rows(); vi++) {
			f_colors.row(vi) = CoolColor::color(0);
		}

		// edge_v1, edge_v2 -> edge_id (order invariant)
		for (int ei = 0; ei < surface_edges.rows(); ei++) {
			auto& edge = surface_edges.row(ei);
			
			if (surf_edge_id_map.find({ edge(0), edge(1) }) == surf_edge_id_map.end()) {
				surf_edge_id_map.insert({ { edge(0), edge(1) }, ei });
			}
		}

		// tri_id -> edge_id * 3
		surf_tri_eids.resize(surface_tris.rows(), 3);
		for (int ti = 0; ti < surface_tris.rows(); ti++) {
			const auto& tri = surface_tris.row(ti);
			auto& eids = surf_tri_eids.row(ti);
			eids(0) = surf_edge_id_map[{tri(0), tri(1)}];
			eids(1) = surf_edge_id_map[{tri(0), tri(2)}];
			eids(2) = surf_edge_id_map[{tri(1), tri(2)}];
		}
			
		compute_face_color();
	}
	
	MeshObject& get_mesh(size_t i) {
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		return *mesh_list[i];
	}

	void translate(const ADU::Vecf_3& trans, size_t i) {
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		auto& mesh = mesh_list[i];
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			verts.row(vi) += trans;
		}
		mesh_list[i]->m_translate = trans;
	}

	void roatate(const ADU::Matf_33& rotate, size_t i) {
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		auto& mesh = mesh_list[i];
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			verts.row(vi) = rotate * verts.row(vi).transpose();
		}
	}

	void roatate(const ADU::Quatf& quat, size_t i) {
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		auto& mesh = mesh_list[i];
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			verts.row(vi) = quat * verts.row(vi).transpose();
		}
		mesh_list[i]->m_rotate = quat;
	}

	void transform(const ADU::Matf_44& tf, size_t i) {
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		auto& mesh = mesh_list[i];
		ADU::Vecf_4 va;
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			va.setZero();
			va.block<3, 1>(0, 0) = verts.row(vi).transpose();
			va(3) = 1;
			va = tf * va;
			va /= va(3);
			verts.row(vi) = va.block<3, 1>(0, 0).transpose();
		}
	}

	void scale(ADU::Real sc, size_t i) { // before precompute geometry info !!!
		using namespace ADU;
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		auto& mesh = mesh_list[i];
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			verts.row(vi) *= sc;
		}
		mesh_list[i]->m_scale = sc;
	}

	void scale(ADU::Real sc_x, ADU::Real sc_y, ADU::Real sc_z, size_t i) { // before precompute geometry info !!!
		using namespace ADU;
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData get_mesh: invalid idx");
		auto& mesh = mesh_list[i];
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			verts.row(vi).x() *= sc_x;
			verts.row(vi).y() *= sc_y;
			verts.row(vi).z() *= sc_z;
		}
		mesh_list[i]->m_scale_x = sc_x;
		mesh_list[i]->m_scale_y = sc_y;
		mesh_list[i]->m_scale_z = sc_z;
	}

	void centerlize(size_t i) {
		using namespace ADU;
		if (i > mesh_list.size()) throw std::runtime_error("Error MeshData centerlize: invalid idx");
		auto& mesh = mesh_list[i];
		Vecf_3 center = Vecf_3::Zero();
		int n_vert = mesh->end_vertIdx - mesh->start_vertIdx;
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			center += verts.row(vi) / static_cast<Real>(n_vert);
		}
		for (int vi = mesh->start_vertIdx; vi < mesh->end_vertIdx; vi++) {
			verts.row(vi) -= center;
		}
	}

	bool on_boundary(int vi) {
		return m_boundary_vinds_set.find(vi) != m_boundary_vinds_set.end();
	}

	const std::vector<int>& get_adjacency(int vi) const {
		if (vi >= adjacent_lists.size()) throw std::runtime_error("Error MeshData get_adjacency: tet type or invalid vi");
		return adjacent_lists[vi];
	}

	void get_mass(ADU::Real density, ADU::Vecf_X& mass) {
		using namespace ADU;

		mass.resize(verts.rows());
		mass.setZero();
		for (int f = 0; f < faces.rows(); ++f) {
			const Veci_3& vinds = faces.row(f);
			Real a = area(f);
			Real tri_mass = density * a;
			mass(vinds(0)) += tri_mass / 3._r;
			mass(vinds(1)) += tri_mass / 3._r;
			mass(vinds(2)) += tri_mass / 3._r;
		}
		for (int ti = 0; ti < tets.rows(); ti++) {
			const Veci_4& vinds = tets.row(ti);
			Real m = volume(ti) * density / 4.0_r;
			mass(vinds(0)) += m;
			mass(vinds(1)) += m;
			mass(vinds(2)) += m;
			mass(vinds(3)) += m;
		}
	}

	void get_mass(ADU::Vecf_X& mass) {
		using namespace ADU;

		mass.resize(verts.rows());
		mass.setZero();
		for (int f = 0; f < faces.rows(); ++f) {
			const Veci_3& vinds = faces.row(f);
			Real a = area(f);
			mass(vinds(0)) += a / 3._r * verts_density(vinds(0));
			mass(vinds(1)) += a / 3._r * verts_density(vinds(1));
			mass(vinds(2)) += a / 3._r * verts_density(vinds(2));
		}
		for (int ti = 0; ti < tets.rows(); ti++) {
			const Veci_4& vinds = tets.row(ti);
			Real v = volume(ti) / 4.0_r;
			mass(vinds(0)) += v * verts_density(vinds(0));
			mass(vinds(1)) += v * verts_density(vinds(1));
			mass(vinds(2)) += v * verts_density(vinds(2));
			mass(vinds(3)) += v * verts_density(vinds(3));
		}
	}

	

	

private:
	std::unordered_set<int> m_boundary_vinds_set;

	void compute_face_color() {
		f_colors.resize(surface_tris.rows(), 3);
		for (auto& mesh : mesh_list) {

			ADU::Vecf_3 color = CoolColor::color(mesh->color_id);
			for (int i = mesh->start_surftriIdx; i < mesh->end_surftriIdx; i++) {
				f_colors.row(i) = color;
			}
		}
	}

	void process_triangle_mesh(const std::string& file_path, MeshObject& mesh, ADU::Real density) {
		using namespace ADU;

		ADU::Matf_X3 temp_verts;
		ADU::Mati_X3 temp_faces;
		ADU::Mati_X2 temp_edges;

		igl::readOBJ(file_path, temp_verts, temp_faces);
		igl::edges(temp_faces, temp_edges);

		size_t start_vidx = verts.rows();
		size_t start_faceidx = faces.rows();

		mesh.start_vertIdx = start_vidx;
		mesh.end_vertIdx = start_vidx + temp_verts.rows();

		mesh.start_faceIdx = start_faceidx;
		mesh.end_faceIdx = start_faceidx + temp_faces.rows();

		mesh.start_tetIdx = mesh.end_tetIdx = 0;

		verts.conservativeResize(verts.rows() + temp_verts.rows(), 3);
		for (size_t i = 0; i < temp_verts.rows(); i++) {
			verts.row(i + mesh.start_vertIdx) = temp_verts.row(i);
		}

		verts_density.conservativeResize(verts_density.rows() + temp_verts.rows(), 1);
		for (size_t i = 0; i < temp_verts.rows(); i++) {
			verts_density(i + mesh.start_vertIdx) = density;
		}

		faces.conservativeResize(faces.rows() + temp_faces.rows(), 3);
		for (size_t i = 0; i < temp_faces.rows(); i++) {
			ADU::Veci_3 f = temp_faces.row(i);
			f.x() += (int)start_vidx;
			f.y() += (int)start_vidx;
			f.z() += (int)start_vidx;
			faces.row(i + mesh.start_faceIdx) = f;
		}

		mesh.start_surftriIdx = surface_tris.rows();
		surface_tris.conservativeResize(surface_tris.rows() + temp_faces.rows(), 3);
		for (size_t i = 0; i < temp_faces.rows(); i++) {
			ADU::Veci_3 f = temp_faces.row(i);
			f.x() += (int)start_vidx;
			f.y() += (int)start_vidx;
			f.z() += (int)start_vidx;
			surface_tris.row(i + mesh.start_surftriIdx) = f;
		}
		mesh.end_surftriIdx = surface_tris.rows();
	}

	void process_tetrahedral_mesh(const std::string& file_path, MeshObject& mesh, ADU::Real density) {
		Eigen::MatrixX<double> temp_pos_d;
		ADU::Matf_XX temp_pos;
		ADU::Mati_XX temp_tets;
		ADU::Mati_XX temp_boundary_facets;
		MeshIO::readTetMesh(file_path, temp_pos_d, temp_tets, temp_boundary_facets, true);
		temp_pos = temp_pos_d.cast<ADU::Real>();

		size_t start_vidx = verts.rows();
		size_t start_tetidx = tets.rows();

		mesh.start_vertIdx = start_vidx;
		mesh.end_vertIdx = start_vidx + temp_pos.rows();

		mesh.start_tetIdx = start_tetidx;
		mesh.end_tetIdx = start_tetidx + temp_tets.rows();

		mesh.start_faceIdx = mesh.end_faceIdx = 0;

		verts.conservativeResize(verts.rows() + temp_pos.rows(), 3);
		for (size_t i = 0; i < temp_pos.rows(); i++) {
			verts.row(i + mesh.start_vertIdx) = temp_pos.row(i);
		}

		verts_density.conservativeResize(verts_density.rows() + temp_pos.rows(), 1);
		for (size_t i = 0; i < temp_pos.rows(); i++) {
			verts_density(i + mesh.start_vertIdx) = density;
		}

		tets.conservativeResize(tets.rows() + temp_tets.rows(), 4);
		for (size_t i = 0; i < temp_tets.rows(); i++) {
			ADU::Veci_4 f = temp_tets.row(i);
			f(0) += (int)start_vidx;
			f(1) += (int)start_vidx;
			f(2) += (int)start_vidx;
			f(3) += (int)start_vidx;
			tets.row(i + mesh.start_tetIdx) = f;
		}

		mesh.start_surftriIdx = surface_tris.rows();
		surface_tris.conservativeResize(surface_tris.rows() + temp_boundary_facets.rows(), 3);
		for (size_t i = 0; i < temp_boundary_facets.rows(); i++) {
			ADU::Veci_3 f = temp_boundary_facets.row(i);
			f.x() += (int)start_vidx;
			f.y() += (int)start_vidx;
			f.z() += (int)start_vidx;
			surface_tris.row(i + mesh.start_surftriIdx) = ADU::Veci_3(f.x(), f.z(), f.y());
		}
		mesh.end_surftriIdx = surface_tris.rows();

	}


};