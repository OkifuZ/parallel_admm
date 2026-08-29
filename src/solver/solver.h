#pragma once

#include "mutils/common_types.h"
#include "contact/contact.h"
#include "constraint/constraint.h"
#include "constraint/nodal_collision_constraint.h"
#include "contact/narrow_phase.h"
#include "animator.h"

#include <Eigen/core>
#include <fstream>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "contact/collision_detection_cuda.h"

//#define PROFILE_RR


class Solver {

	
public:

	/// Per-step performance/contact metrics, collected by each solver's step()
	/// and written as CSV by write_step_metrics_csv() (comparison tooling).
	struct StepMetrics {
		size_t frame{};
		double step_ms{};
		size_t n_contacts{};
	};
	std::vector<StepMetrics> step_metrics;
	size_t step_cnt = 0;

	/// Write step_metrics as CSV (header: frame,step_ms,n_contacts).
	void write_step_metrics_csv(const std::string& path) const {
		std::ofstream f(path);
		if (!f.is_open()) return;
		f << "frame,step_ms,n_contacts\n";
		for (const auto& m : step_metrics) {
			f << m.frame << "," << m.step_ms << "," << m.n_contacts << "\n";
		}
	}

	using ConstraintsList = std::vector<std::shared_ptr<Constraint>>;
	using NodalCollisionConstraintsList = std::vector<std::shared_ptr<NodalCollisionConstraint>>;
	using ObstacleList = std::vector<std::shared_ptr<Obstacle>>;
	using ObstacleContactInfoList = std::vector<std::shared_ptr<ObstacleContactInfo>>;

	std::unique_ptr<ScriptAnimator> animator;

	BVH_GPU* bvh;

	ADU::Real g{ static_cast <ADU::Real>(0.98) };

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
		//m_dt /= admm_max_iter;
		m_dt2 = m_dt * m_dt;
		m_dt_inv = 1.0_r / m_dt;
		m_nVert = verts.rows();
		init();
	}

	virtual void reset(const ADU::Matf_X3& ini_verts, bool need_precompute = false) {
		
	}

	virtual void add_constraints(const ConstraintsList& constraints_) {
		this->m_constraints.insert(this->m_constraints.end(), constraints_.begin(), constraints_.end());

		/*std::cout << "constraints: ";
		for (auto& constraint : this->m_constraints) {
			std::cout << constraint->type << " ";
		}
		std::cout << std::endl;*/
	}

	virtual void add_nodal_constraints(const NodalCollisionConstraintsList& constraints_) {
		m_nodal_collision_constraint_start = m_constraints.size();
		for (const auto& ct : constraints_) {
			m_vInd_cInd.insert({ct->inds[0], m_constraints.size()});
			m_constraints.emplace_back(ct);
		}
		m_nodal_collision_constraint_end = m_constraints.size();
		m_enable_nodal_obstacle_collision = true;
	}

	virtual void step() {
		using namespace ADU;
		for (int i = 0; i < m_nVert; i++) {
			m_vertices.row(i) -= ADU::Vecf_3{0, 0.01_r, 0};
		}
	}

	virtual const ADU::Matf_X3& getVertices() const {
		return m_vertices;
	}

	virtual ADU::Matf_X3& getVertices() {
		return m_vertices;
	}

	virtual const ADU::Matf_X3& getVelocities() const {
		return m_velocities;
	}

	virtual ADU::Matf_X3& getVelocities() {
		return m_velocities;
	}

	ConstraintsList& getConstraints() {
		return m_constraints;
	}

	const ConstraintsList& getConstraints() const {
		return m_constraints;
	}

	virtual void addPins(const std::vector<int>& pin_inds) {
		m_pin_inds_set.insert(pin_inds.begin(), pin_inds.end());
	}

	virtual void addObstacle(const std::shared_ptr<Obstacle>& obstacle) {
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

	void animate_step();

	ADU::Matf_X3 m_vertices;
	ADU::Matf_X3 m_velocities;
	ADU::Vecf_X m_M_vec;
	ADU::Vecf_X m_M_inv_vec;

	std::unordered_set<int> m_pin_inds_set;

	ConstraintsList m_constraints;
	int m_nodal_collision_constraint_start{};
	int m_nodal_collision_constraint_end{};
	bool m_enable_nodal_obstacle_collision{ false };

	ObstacleList m_obstacles;

	std::unordered_map<int, int> m_vInd_cInd;

	ObstacleContactInfoList m_obsContact_infolist;

	ADU::Real m_dt{};
	ADU::Real m_dt2{};
	ADU::Real m_dt_inv{};
	int m_nVert{};
protected:

};