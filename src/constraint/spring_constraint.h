#pragma once
#include "constraint/constraint.h"
#include "mutils/common_types.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <iostream>
#include <zensim/math/VecInterface.hpp>

class SpringConstraint : public Constraint {
public:

	ADU::Real rest_length{};

	SpringConstraint() = delete;

	SpringConstraint(int vinda,int vindb, ADU::Matf_X3& verts,
		ADU::Real k, int global_idx, int start_row) :
		Constraint() {
		using namespace ADU;

		type = 6;
		this->k = k;
		this->global_m = global_idx;
		this->start_row = start_row;

		inds.resize(1, 0);
		inds[0] = vinda;
		inds[1] = vindb;
		dim = 2;

		pin_poistion = verts.row(vind).transpose();
		
		w = std::sqrt(k);
	}

	virtual void get_D(std::vector< ADU::Tripf >& triplets, bool local = true) const {
		// D \in R^{1xn}
		using namespace ADU;

		// TODO
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

		Vecf_3& v0 = zi.row(0);
		Vecf_3& v1 = zi.row(1);
		Vecf_3 v2 = v0 - v1;
		Real dist = std::sqrt(v2.dot(v2));
		Vecf_3 center = 0.5_r * (v0 + v1);

		v0 = center + 0.5f * rest_length * (v0 - center) / dist;
		v1 = center + 0.5f * rest_length * (v1 - center) / dist;

	};


};
