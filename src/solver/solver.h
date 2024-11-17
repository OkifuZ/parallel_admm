#pragma once

#include "mutils/common_types.h"
#include "contact/contact.h"
#include "constraint/constraint.h"
#include "constraint/nodal_collision_constraint.h"
#include "constraint/XPBD_constraints.h"
#include "contact/narrow_phase.h"
#include "animator.h"

#include <Eigen/core>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "contact/collision_detection_cuda.h"

//#define PROFILE_RR


class Solver {

	
public:


	using XPBDConstraintList = std::vector<std::shared_ptr<XPBDConstraint>>;
	using ConstraintsList = std::vector<std::shared_ptr<Constraint>>;
	using NodalCollisionConstraintsList = std::vector<std::shared_ptr<NodalCollisionConstraint>>;
	using ObstacleList = std::vector<std::shared_ptr<Obstacle>>;
	using ObstacleContactInfoList = std::vector<std::shared_ptr<ObstacleContactInfo>>;

	std::unique_ptr<ScriptAnimator> animator;

	BVH_GPU* bvh;

	ADU::Real g{ static_cast <ADU::Real>(0.98) };


	int m_nCDim{};
	bool parallel = true;

	std::unique_ptr<ProximalQuery> prox_query{ nullptr };

	Solver(){
		m_obsContact_infolist.reserve(256);
	}

	virtual void init() {}

	virtual void init(const ADU::Matf_X3& verts, const ADU::Vecf_X& mass, ADU::Real dt) {
		using namespace ADU;
		if (m_constraints.empty()) {
			make_exception("solver init encountered empty constraints");
		}

		m_vertices = verts;
		m_velocities.resizeLike(m_vertices);
		m_velocities.setZero();
		m_M_vec = mass;
		m_M_inv_vec = mass;
		for (int i = 0; i < m_M_inv_vec.rows(); i++) m_M_inv_vec(i) = 1.0_r / m_M_vec(i);

		for (const auto& ct : m_constraints) {
			if (ct->type == 4) m_M_inv_vec(ct->inds[0]) = 0;
		}

		m_dt = dt;
		m_dt2 = m_dt * m_dt;
		m_dt_inv = 1.0_r / m_dt;
		m_nVert = verts.rows();
		init();
	}

	virtual void reset(const ADU::Matf_X3& ini_verts, bool need_precompute = false) {
		
	}

	void add_constraints(const ConstraintsList& constraints_) {
		this->m_constraints.insert(this->m_constraints.end(), constraints_.begin(), constraints_.end());

		/*std::cout << "constraints: ";
		for (auto& constraint : this->m_constraints) {
			std::cout << constraint->type << " ";
		}
		std::cout << std::endl;*/
	}

	void add_nodal_constraints(const NodalCollisionConstraintsList& constraints_) {
		m_nodal_collision_constraint_start = m_constraints.size();
		for (const auto& ct : constraints_) {
			m_vInd_cInd.insert({ct->inds[0], m_constraints.size()});
			m_constraints.emplace_back(ct);
		}
		m_nodal_collision_constraint_end = m_constraints.size();
		m_enable_nodal_obstacle_collision = true;
	}

	void add_XPBDConstraints(const XPBDConstraintList& constraints_) {
		this->m_XPBDconstraints.insert(this->m_XPBDconstraints.end(), constraints_.begin(), constraints_.end());
	}

	virtual void step() {
		using namespace ADU;
		for (int i = 0; i < m_nVert; i++) {
			m_vertices.row(i) -= ADU::Vecf_3{0, 0.01_r, 0};
		}
	}

	const ADU::Matf_X3& getVertices() const {
		return m_vertices;
	}

	ADU::Matf_X3& getVertices() {
		return m_vertices;
	}

	const ADU::Matf_X3& getVelocities() const {
		return m_velocities;
	}

	ADU::Matf_X3& getVelocities() {
		return m_velocities;
	}

	ConstraintsList& getConstraints() {
		return m_constraints;
	}

	const ConstraintsList& getConstraints() const {
		return m_constraints;
	}

	void addPins(const std::vector<int>& pin_inds) {
		m_pin_inds_set.insert(pin_inds.begin(), pin_inds.end());
	}

	void addObstacle(const std::shared_ptr<Obstacle>& obstacle) {
		m_obstacles.push_back(obstacle);
	}

	void clearObstacles() {
		m_obstacles.clear();
	}

	ObstacleList& getObstacles() {
		return m_obstacles;
	}

	const ObstacleList& getObstacles() const {
		return m_obstacles;
	}

	ADU::Veci_X m_vStatic;

	void animate_step();

	ADU::Matf_X3 m_vertices;
	ADU::Matf_X3 m_velocities;
	ADU::Vecf_X m_M_vec;
	ADU::Vecf_X m_M_inv_vec;
	ADU::SpMatf m_M_bar_vec;
	ADU::Vecf_X m_M_bar_inv_vec;

	std::unordered_set<int> m_pin_inds_set;

	ConstraintsList m_constraints;
	int m_nodal_collision_constraint_start{};
	int m_nodal_collision_constraint_end{};
	bool m_enable_nodal_obstacle_collision{ false };

	XPBDConstraintList m_XPBDconstraints;

	ObstacleList m_obstacles;

	std::unordered_map<int, int> m_vInd_cInd;

	ObstacleContactInfoList m_obsContact_infolist;

	ADU::Real m_dt{};
	ADU::Real m_dt2{};
	ADU::Real m_dt_inv{};
	int m_nVert{};
protected:

};