#pragma once
#include "constraint/constraint.h"
#include "mutils/common_types.h"
#include "constraint/XPBD_constraints.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <iostream>

class TriangleConstraint : public Constraint {
public:
	ADU::Matf_22 Binv{};
	ADU::Matf_22 Binv_XPBD{};
	ADU::Real area{};
	ADU::Real min_limit{};
	ADU::Real max_limit{};
	bool use_limit{ true };


	TriangleConstraint() = delete;

	TriangleConstraint(const Eigen::Vector3i& vinds, const ADU::Matf_X3& verts,
		ADU::Real min_limit, ADU::Real max_limit, ADU::Real k, bool use_limit, int global_idx, int start_row) :
		Constraint(), min_limit(min_limit), max_limit(max_limit), use_limit(use_limit) {
		using namespace ADU;

		type = 1;
		this->k = k;
		this->global_m = global_idx;
		this->start_row = start_row;

		if (min_limit > 1.0_r || min_limit <= 0.0_r || max_limit <= 1.0_r) {
			throw std::runtime_error("TriangleConstraint Error: Invalid strain limiting");
		}

		inds.resize(3, 0);
		inds[0] = vinds[0];
		inds[1] = vinds[1];
		inds[2] = vinds[2];
		dim = 2;

		Vecf_3 e12 = verts.row(inds[1]) - verts.row(inds[0]);
		Vecf_3 e13 = verts.row(inds[2]) - verts.row(inds[0]);
		Vecf_3 n1 = e12.normalized();
		Vecf_3 n2 = (e13 - e13.dot(n1) * n1).normalized();
		Matf_32 basis;
		Matf_32 edges;
		basis.col(0) = n1; basis.col(1) = n2;
		edges.col(0) = e12; edges.col(1) = e13;
		Binv = (basis.transpose() * edges).inverse(); // Rest pose matrix

		area = (basis.transpose() * edges).determinant() / 2.0_r;

		if (area < 0) {
			throw std::runtime_error("TriangleConstraint Error: Inverted initial pose");
		}

		w = std::sqrt(k * area);

	}

	virtual void get_D(std::vector< ADU::Tripf >& triplets, bool local = true) const {
		// D \in R^{2xn}
		using namespace ADU;

		Matf_32 S;
		S.setZero();
		S(0, 0) = -1;
		S(0, 1) = -1;
		S(1, 0) = 1;
		S(2, 1) = 1;

		// F^T = G^T * J * X
		// G^T is ommitted here
		//printf("tri: ");
		Matf_32 G = S * Binv;
		int cols[3] = { inds[0], inds[1], inds[2] };
		if (local) {
			for (int j = 0; j < 3; ++j) {
				triplets.emplace_back(0, cols[j], G(j, 0));
				triplets.emplace_back(1, cols[j], G(j, 1));
			}
		}
		else {
			for (int j = 0; j < 3; ++j) {
				triplets.emplace_back(start_row + 0, cols[j], G(j, 0));
				triplets.emplace_back(start_row + 1, cols[j], G(j, 1));
				//printf("%f %f ", G(j, 0), G(j, 1));
			}
			//printf("\n");
		}
		
	}

	virtual void test_D(const ADU::Matf_X3& pos) const {
		using namespace ADU;

		SpMatf Di(2, pos.rows());
		std::vector< Tripf > triplets;
		get_D(triplets);
		Di.setFromTriplets(triplets.begin(), triplets.end());
		Matf_32 F_s = (Di * pos).transpose();

		Vecf_3 e12 = pos.row(inds[1]) - pos.row(inds[0]);
		Vecf_3 e13 = pos.row(inds[2]) - pos.row(inds[0]);
		Matf_32 F_a;
		F_a.col(0) = e12;
		F_a.col(1) = e13;
		F_a *= Binv;

		Real error = (F_a - F_s).norm();
		if (error > 1e-7_r) {
			printf("test failed!!!\n");
			std::cout << global_m << ": " << inds[0] << " " << inds[1] << " " << inds[2] << "\n";
			std::cout << F_s << "\n";
			std::cout << F_a << "\n*****************\n\n";
		}
		else {
			//printf("D pass test\n");
		}

	}


	virtual void prox(ADU::Matf_XX& zi) {
		using namespace Eigen;
		using namespace ADU;
		//std::cout << "prox() row norm: " << zi.row(0).norm() << " " << zi.row(1).norm() << "\n";

		Matf_32 F = zi.transpose();
		JacobiSVD<Matf_32 > svd(F, ComputeFullU | ComputeFullV);
		Matf_32 S = Matf_32::Zero();
		S.block<2, 2>(0, 0) = Matf_22::Identity();
		Matf_32 P = svd.matrixU() * S * svd.matrixV().transpose();
		zi = (0.5_r * (P + F)).transpose();

		// If w^2 != k*volume, use this:
		// double k = lame.bulk_modulus();
		// double ka = k * area;
		// double w2 = weight*weight;
		// zi = (ka*p + w2*zi) / (w2 + ka);

		if (use_limit) {
			const bool check_strain = min_limit > 0.0_r || max_limit < 99.0_r;
			if (check_strain) {
				Real l_col0 = zi.row(0).norm();
				Real l_col1 = zi.row(1).norm();
				if (l_col0 < min_limit) { zi.row(0) *= (min_limit / l_col0); }
				if (l_col1 < min_limit) { zi.row(1) *= (min_limit / l_col1); }
				if (l_col0 > max_limit) { zi.row(0) *= (max_limit / l_col0); }
				if (l_col1 > max_limit) { zi.row(1) *= (max_limit / l_col1); }
			}
		}
	};

	virtual void get_F_shape(int& r, int& c) {
		r = 3;
		c = 2;
	}

	virtual ADU::Real get_E(const ADU::Vecf_X& pos) const {
		// 0.5 * k * C^2
		return 0;
	} 

	virtual void get_Grad_C(const ADU::Matf_X3& pos, ADU::Vecf_X& grad) const {
		using namespace ADU;
		const unsigned i1 = inds[0];
		const unsigned i2 = inds[1];
		const unsigned i3 = inds[2];

		auto& x1 = pos.row(i1);
		auto& x2 = pos.row(i2);
		auto& x3 = pos.row(i3);

		std::vector<Vecf_3> grads;
		Real c{};
		bool res = XPBD_utils::get_FEMTriangleGradient(
			x1, x2, x3, area, Binv_XPBD, 
			k, k, k, 0.3_r, 0.3_r,
			grads, c
		);
		std::cout << c << "\n";

		grad.block<3, 1>(i1 * 3, 0) += grads[0];
		grad.block<3, 1>(i2 * 3, 0) += grads[1];
		grad.block<3, 1>(i3 * 3, 0) += grads[2];
	}

	virtual void get_Grad_U(const ADU::Matf_X3& pos, ADU::Vecf_X& grad) const {
		using namespace ADU;
		const unsigned i1 = inds[0];
		const unsigned i2 = inds[1];
		const unsigned i3 = inds[2];

		auto& x1 = pos.row(i1);
		auto& x2 = pos.row(i2);
		auto& x3 = pos.row(i3);

		std::vector<Vecf_3> grads;
		Real c{};
		bool res = XPBD_utils::get_FEMTriangleGradient(
			x1, x2, x3, area, Binv_XPBD,
			k, k, k, 0.3_r, 0.3_r, 
			grads, c
		);

		grad.block<3, 1>(i1 * 3, 0) += grads[0] * c;
		grad.block<3, 1>(i2 * 3, 0) += grads[1] * c;
		grad.block<3, 1>(i3 * 3, 0) += grads[2] * c;
	}

};