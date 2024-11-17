#pragma once

#include "mutils/common_types.h"
#include "XPBD_constraints.h"
#include <mutils/common_type_hostonly.h>

#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>


static int total_ctype = 5;


class Constraint {
public:

	std::vector<int> inds{};
	int type = 0; // 1: triangle, 2: tetrahedral, 3: bending, 4: pin, 5: collision 6. spring

	ADU::Real k{};
	ADU::Real w{};

	int dim = 0;
	int start_row{};
	int global_m{};

	//std::shared_ptr<XPBDConstraint> XPBD_ct;

	Constraint() {}

	virtual void get_D(std::vector< ADU::Tripf >& triplets, bool local = true) const {}

	virtual void test_D(const ADU::Matf_X3& pos) const {}

	virtual void prox(ADU::Matf_XX& zi) {};

	virtual void get_F_shape(int& r, int& c) {}

	virtual ADU::Real get_E(const ADU::Vecf_X& pos) const { return 0; };
	// grad_C * sqrt(k)
	virtual void get_Grad_C(const ADU::Matf_X3& pos, ADU::Vecf_X& grad) const {};
	virtual void get_Grad_U(const ADU::Matf_X3& pos, ADU::Vecf_X& grad) const {};
};

