#pragma once

#include "constraint/constraint.h"
#include "mutils/common_types.h"
#include "mutils/cformat.h"
#include "mutils/exception_handle.h"

#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>


class BendingConstraint : public Constraint {
public:

	ADU::Vecf_3 rest_mean_curvature{};
	ADU::Real rest_mean_curvature_norm{};
	std::vector<ADU::Real> coeffs; // middle vert, ring verts
	ADU::Real varea;

	BendingConstraint() = delete;

	BendingConstraint(int middle_idx, const std::vector<int>& ring_inds, const ADU::Matf_X3& verts,
		ADU::Real k, int global_idx, int start_row) :
		Constraint() {
		using namespace ADU;

		if (ring_inds.size() < 3) {
			;
			throw std::runtime_error(ADU::cformat("v %d: BendingConstraint Error: ring num < 3", middle_idx));
		}

		type = 3;
		this->k = k;
		this->global_m = global_idx;
		this->start_row = start_row;

		inds.push_back(middle_idx);
		inds.insert(inds.end(), ring_inds.begin(), ring_inds.end());
		dim = 1;

		// initial mean curvature at middle vertex
		getLaplaceBeltramiDiscretisationMeanValueCoefficients(
			middle_idx, ring_inds, verts, coeffs);
		Real middle_coef = 0.0f;
		for (int i = 0; i < coeffs.size(); i++) {
			middle_coef += coeffs[i];
			coeffs[i] *= -1.0_r;
		}
		coeffs.insert(coeffs.begin(), middle_coef);

		rest_mean_curvature.setZero();
		for (int i = 0; i < coeffs.size(); i++) {
			rest_mean_curvature += coeffs[i] * verts.row(inds[i]);
		}
		rest_mean_curvature_norm = rest_mean_curvature.norm();
		if (rest_mean_curvature_norm < 1e-7_r) rest_mean_curvature_norm = 0;

		rest_mean_curvature_norm = 0.0_r;
		
		// area
		const Vecf_3& v_middle = verts.row(middle_idx);
		Real radius{ 0.0_r };
 		for (int i = 0; i < ring_inds.size(); i++) {
			const Vecf_3& v = verts.row(ring_inds[i]);
			radius += (v - v_middle).norm();
		}
		radius /= ring_inds.size();
		varea = 3.1415926_r * radius * radius / 3.0_r;
		w = std::sqrt(k * varea);
	}

	static ADU::Real getSine(const ADU::Vecf_3& v1, const ADU::Vecf_3& v2, const ADU::Vecf_3& v3)
	{
		ADU::Vecf_3 edge1 = v1 - v2;
		ADU::Vecf_3 edge2 = v3 - v2;
		return edge1.cross(edge2).norm() / (edge1.norm() * edge2.norm());
	}

	static ADU::Real getCosine(const ADU::Vecf_3& v1, const ADU::Vecf_3& v2, const ADU::Vecf_3& v3)
	{
		ADU::Vecf_3 edge1 = v1 - v2;
		ADU::Vecf_3 edge2 = v3 - v2;
		return edge1.dot(edge2) / (edge1.norm() * edge2.norm());
	}

	static ADU::Real getMeanValue(const ADU::Vecf_3& v1, const ADU::Vecf_3& v2, const ADU::Vecf_3& v3)
	{
		// tan a/2 = (1-cos a) / sin a = sin a / (1 + cos a)
		ADU::Real numerator = (1 - getCosine(v1, v2, v3));
		if (numerator == 0)
		{
			return 0;
		}
		return numerator / getSine(v1, v2, v3);
	}

	static bool co_linear(const ADU::Vecf_3& v1, const ADU::Vecf_3& v2, const ADU::Vecf_3& v3) {
		if ((v1 - v2).cross(v3 - v2).squaredNorm() < 1e-9) return true;
		return false;
	}

	static void getLaplaceBeltramiDiscretisationMeanValueCoefficients(
		int middle_idx, const std::vector<int>& ring_inds, const ADU::Matf_X3& verts, 
		std::vector<ADU::Real>& coeffs) 
	{
		using namespace ADU;
		coeffs.clear();
		const Vecf_3& middle_vertex = verts.row(middle_idx);
		int ring_num = ring_inds.size();
		Real distance{};
		Real clockwise_mean_value{}, counter_clockwise_mean_value{};
		Real final_value{};
		if (ring_num <= 2) ADU::make_exception(ADU::cformat("v %d: bending constraint ring num<=2 occured!", middle_idx));

		for (int i = 0; i < ring_num; i++) {
			const Vecf_3& vertex = verts.row(ring_inds[i]);
			const Vecf_3& clockwise_vertex = verts.row(ring_inds[(i + 1) % ring_num]);
			const Vecf_3& counter_clockwise_vertex = verts.row(ring_inds[(i + ring_num - 1) % ring_num]);

			// We can't have a flat triangle around the middle vertex. If that
			// was the case, the triangulation of the mesh would be wrong or
			// the middle vertex would be on a border.
			// TODO colinear check
			/*assert(!areCollinear(vertex, clockwise_vertex, middle_vertex));
			assert(!areCollinear(vertex, middle_vertex, clockwise_vertex));*/

			distance = (vertex - middle_vertex).norm();
			final_value = 0.0_r;
			if (!co_linear(vertex, middle_vertex, clockwise_vertex)) {
				clockwise_mean_value = getMeanValue(vertex, middle_vertex, clockwise_vertex);
				final_value += clockwise_mean_value;
			}
			if (!co_linear(vertex, middle_vertex, counter_clockwise_vertex)) {
				counter_clockwise_mean_value =
					getMeanValue(vertex, middle_vertex, counter_clockwise_vertex);
				final_value += counter_clockwise_mean_value;
			}
			if (final_value == 0.0_r) {
				printf("%d, ", middle_idx);
				ADU::make_exception(ADU::cformat("v %d: flat tri in bending constraints", middle_vertex));
			}

			coeffs.push_back(final_value / distance);
		}
	}

	virtual void get_D(std::vector< ADU::Tripf >& triplets, bool local = true) const {
		// D \in R^{1xn}
		using namespace ADU;

		if (local) {
			for (int i = 0; i < inds.size(); i++) {
				triplets.emplace_back(0, inds[i], coeffs[i]);
			}
		}
		else {
			for (int i = 0; i < inds.size(); i++) {
				triplets.emplace_back(start_row, inds[i], coeffs[i]);
			}
		}
		
	}


	virtual void prox(ADU::Matf_XX& zi) {
		using namespace Eigen;
		using namespace ADU;

		ADU::Vecf_3 z = zi.transpose();
		ADU::Vecf_3 p = z;
		Real pnorm = p.norm();

		if (rest_mean_curvature_norm == 0) {
			p = Vecf_3::Zero();
		}
		else if (pnorm > 1e-7_r) {
			p *= rest_mean_curvature_norm / pnorm;
		}
		else { // znorm -> 0
			// TODO
			p = Vecf_3::Zero();
		}

		zi = (z + p).transpose() / 2.0_r;
		/*Real w2 = w * w;
		zi = (k * p + w2 * z) / (w2 + k);*/
	};

};