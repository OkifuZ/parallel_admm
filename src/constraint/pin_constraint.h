#pragma once
#include "constraint/constraint.h"
#include "mutils/common_types.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <iostream>

class PinConstraint : public Constraint {
public:

	ADU::Vecf_3 pin_poistion{};

	PinConstraint() = delete;

	PinConstraint(int vind, ADU::Matf_X3& verts,
		ADU::Real k, int global_idx, int start_row) :
		Constraint() {
		using namespace ADU;

		type = 4;
		this->k = k;
		this->global_m = global_idx;
		this->start_row = start_row;

		inds.resize(1, 0);
		inds[0] = vind;
		dim = 1;

		pin_poistion = verts.row(vind).transpose();
		
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

		zi = pin_poistion.transpose();
		
	};

};