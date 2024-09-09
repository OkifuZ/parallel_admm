#pragma once
#include "constraint/constraint.h"
#include "mutils/common_types.h"
#include "contact/contact.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <iostream>

class NodalCollisionConstraint : public Constraint {
public:

	bool active{ false };
	std::shared_ptr<ObstacleContactInfo> contact_info{nullptr};
	ADU::Vecf_3 x_0;

	NodalCollisionConstraint() = delete;

	NodalCollisionConstraint(int vind, ADU::Matf_X3& verts,
		ADU::Real k, int global_idx, int start_row) :
		Constraint() {
		using namespace ADU;

		type = 5;
		this->k = k;
		this->global_m = global_idx;
		this->start_row = start_row;

		inds.resize(1, 0);
		inds[0] = vind;
		dim = 1;

		w = std::sqrt(k);
	}

	virtual void get_D(std::vector< ADU::Tripf >& triplets, bool local = true) const {
		// D \in R^{1xn}
		using namespace ADU;

		if (local) {
			triplets.emplace_back(0, inds[0], 1);
		}
		else {
			triplets.emplace_back(start_row, inds[0], 1);
		}
	}


	virtual void prox(ADU::Matf_XX& zi) {
		using namespace Eigen;
		using namespace ADU;

		if (!active) return;

		Vecf_3 z = zi.transpose();

		// we can not directly use contact_info, because zi = Dx+u
		// the existance of u make zi not the same as contact_info which describe Dx
		int obstacle_type = contact_info->obstacle->type;

		if (obstacle_type == 0) { // sphere
			const auto& sphere = std::static_pointer_cast<SphereObstacle>(contact_info->obstacle);
			Vecf_3 zmc = z - sphere->center;
			Real dis = zmc.norm();
			if (dis < 1e-6_r) {} // danger, ignore
			else if (dis < sphere->radius) {
				z = zmc / dis * sphere->radius + sphere->center;
			}
		}
		else if (obstacle_type == 1) { // plane
			const auto& plane = std::static_pointer_cast<PlaneObstacle>(contact_info->obstacle);
			Vecf_3 zmc = z - plane->center;
			Real penetration = zmc.dot(plane->normal);
			if (penetration < 0) {
				z = z - plane->normal * penetration;
				auto& dn = (z - x_0).dot(plane->normal) * plane->normal;
				auto& dt = (z - x_0) - dn;
				z = x_0 + dn + 0.8 * dt;
			}
		}
		else {
			throw std::runtime_error("Error NodalCollisionConstraint prox: invalid obstacle type");
		}

		zi = z.transpose();
	};

	void activate(const std::shared_ptr<ObstacleContactInfo>& contact_info_, const ADU::Vecf_3& x_0) {
		active = true;
		if (!contact_info_) throw std::runtime_error("Error: NodalCollisionConstraint activate, empty contact_info");
		this->contact_info = contact_info_;
		this->x_0 = x_0;
	}

	void deactivate() {
		active = false;
		contact_info = nullptr;
	}

};