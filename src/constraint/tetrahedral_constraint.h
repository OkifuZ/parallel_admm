#pragma once
#include "constraint/constraint.h"
#include "mutils/common_types.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <iostream>

class TetrahedralConstraint : public Constraint {
public:
	ADU::Matf_33 Binv{};
	ADU::Real volume{};
	ADU::Real min_limit{};
	ADU::Real max_limit{};
	bool use_limit{ false };


	TetrahedralConstraint() = delete;

	TetrahedralConstraint(const Eigen::Vector4i& vinds, const ADU::Matf_X3& verts,
		ADU::Real min_limit, ADU::Real max_limit, bool use_limit, ADU::Real k, int global_idx, int start_row) :
		Constraint(), min_limit(min_limit), max_limit(max_limit), use_limit(use_limit) {
		using namespace ADU;

		type = 2;
		this->k = k;
		this->global_m = global_idx;
		this->start_row = start_row;

		if (min_limit > 1.0_r || min_limit <= 0.0_r || max_limit <= 1.0_r) {
			throw std::runtime_error("TetrahedralConstraint Error: Invalid strain limiting");
		}

		inds.resize(4, 0);
		inds[0] = vinds[0];
		inds[1] = vinds[1];
		inds[2] = vinds[2];
		inds[3] = vinds[3];
		dim = 3;

		Vecf_3 e12 = verts.row(inds[1]) - verts.row(inds[0]);
		Vecf_3 e13 = verts.row(inds[2]) - verts.row(inds[0]);
		Vecf_3 e14 = verts.row(inds[3]) - verts.row(inds[0]);
		Matf_33 B0;
		B0.col(0) = e12;
		B0.col(1) = e13;
		B0.col(2) = e14;
		Binv = B0.inverse(); // Rest pose matrix

		volume = std::abs(B0.determinant() / 6.0_r);

		w = std::sqrt(k * volume);

	}

	virtual void get_D(std::vector< ADU::Tripf >& triplets, bool local = true) const {
		// D \in R^{2xn}
		using namespace ADU;

		Matf_43 S;
		S.setZero();
		S(0, 0) = -1;
		S(0, 1) = -1;
		S(0, 2) = -1;
		S(1, 0) = 1;
		S(2, 1) = 1;
		S(3, 2) = 1;

		// F^T = G^T * J * X
		// G^T is ommitted here
		Matf_43 G = S * Binv;
		int cols[4] = { inds[0], inds[1], inds[2], inds[3] };
		if (local) {
			for (int j = 0; j < 4; ++j) {
				triplets.emplace_back(0, cols[j], G(j, 0));
				triplets.emplace_back(1, cols[j], G(j, 1));
				triplets.emplace_back(2, cols[j], G(j, 2));
			}
		}
		else {
			for (int j = 0; j < 4; ++j) {
				triplets.emplace_back(start_row + 0, cols[j], G(j, 0));
				triplets.emplace_back(start_row + 1, cols[j], G(j, 1));
				triplets.emplace_back(start_row + 2, cols[j], G(j, 2));
			}
		}
	}

	virtual void test_D(const ADU::Matf_X3& pos) const {
		using namespace ADU;

		SpMatf Di(3, pos.rows());
		std::vector< Tripf > triplets;
		get_D(triplets);
		Di.setFromTriplets(triplets.begin(), triplets.end());
		Matf_33 F_s = (Di * pos).transpose();

		Vecf_3 e12 = pos.row(inds[1]) - pos.row(inds[0]);
		Vecf_3 e13 = pos.row(inds[2]) - pos.row(inds[0]);
		Vecf_3 e14 = pos.row(inds[3]) - pos.row(inds[0]);
		Matf_33 B;
		B.col(0) = e12;
		B.col(1) = e13;
		B.col(2) = e14;
		Matf_33 F_a = B * Binv;

		Real error = (F_a - F_s).norm();
		if (error > 1e-7_r) {
			printf("tetrahedral test failed!!!\n");
			std::cout << global_m << ": " << inds[0] << " " << inds[1] << " " << inds[2] << "\n";
			std::cout << F_s << "\n";
			std::cout << F_a << "\n*****************\n\n";
			throw std::runtime_error("Error TetrahedralConstraint: test D failed");
		}
		else {
			//printf("D pass test\n");
		}

	}


	virtual void prox(ADU::Matf_XX& zi) {
		using namespace Eigen;
		using namespace ADU;
		//std::cout << "prox() row norm: " << zi.row(0).norm() << " " << zi.row(1).norm() << "\n";

		Matf_33 F = zi.transpose();
		JacobiSVD<Matf_33 > svd(F, ComputeFullU | ComputeFullV);
		Matf_33 S = Matf_33::Identity();
		if (F.determinant() < 0) S(2, 2) = -1.0_r;
		Matf_33 P = svd.matrixU() * S * svd.matrixV().transpose();
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
				Real l_col2 = zi.row(2).norm();
				if (l_col0 < min_limit) { zi.row(0) *= (min_limit / l_col0); }
				if (l_col1 < min_limit) { zi.row(1) *= (min_limit / l_col1); }
				if (l_col2 < min_limit) { zi.row(2) *= (min_limit / l_col2); }
				if (l_col0 > max_limit) { zi.row(0) *= (max_limit / l_col0); }
				if (l_col1 > max_limit) { zi.row(1) *= (max_limit / l_col1); }
				if (l_col2 > max_limit) { zi.row(2) *= (max_limit / l_col2); }
			}
		}
	};

	virtual void get_F_shape(int& r, int& c) {
		r = 3;
		c = 3;
	}

};