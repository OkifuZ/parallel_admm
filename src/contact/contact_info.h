#pragma once
#include "mutils/common_types.h"
#include "mutils/distEE.h"
#include "mutils/distPT.h"
#include "contact/contact_parameter.h"
#include "contact/broad_phase.h"

#include "mutils/dist_g.cuh"

#include <memory>
#include <vector>



class ContactInfo {

public:
	enum PairType {
		None,
		PT,
		EE
	};


	PairType pair_type{ None };

	std::array<size_t, 4> vinds; // P,T0,T1,T2; E01,E02,E11,E12

	ADU::Vecf_3 normal;
	ADU::Vecf_3 point;

	ADU::Real dist;
	ADU::Real distSqr;

	ADU::Vecf_4 bary;
	ADU::Vecf_4 ini_bary;

	ADU::Real h_cN;

	ADU::Vecf_3 Sa;
	ADU::Vecf_3 Sb;
	ADU::Vecf_3 Sm;
	ADU::Real d;

	// TODO: decouple this
	ADU::Vecf_3 r_c{0, 0, 0};

	friend bool operator == (const ContactInfo& c1, const ContactInfo& c2){
		ADU::Real const diff_x = std::fabs(c1.point.x() - c2.point.x());
		ADU::Real const diff_y = std::fabs(c1.point.y() - c2.point.y());
		ADU::Real const diff_z = std::fabs(c1.point.z() - c2.point.z());
		if (diff_x > ContactParameter::precision) return false;
		if (diff_y > ContactParameter::precision) return false;
		if (diff_z > ContactParameter::precision) return false;

		/*ADU::Real const diff_x_n = std::fabs(c1.normal.x() - c2.normal.x());
		ADU::Real const diff_y_n = std::fabs(c1.normal.y() - c2.normal.y());
		ADU::Real const diff_z_n = std::fabs(c1.normal.z() - c2.normal.z());
		if (diff_x_n > ContactParameter::precision) return false;
		if (diff_y_n > ContactParameter::precision) return false;
		if (diff_z_n > ContactParameter::precision) return false;*/

		return true;
		/*if (c1.pair_type != c2.pair_type) return false;
		return c1.vinds[0] == c2.vinds[0] && c1.vinds[1] == c2.vinds[1] && c1.vinds[2] == c2.vinds[2] && c1.vinds[3] == c2.vinds[3];*/
	}

	ContactInfo() {
		// for std::vector
	};

	ContactInfo(const Result<ADU::Real>& result, size_t v0, size_t v1, size_t v2, size_t v3,
		const ADU::Vecf_3& normal, const ADU::Vecf_3& point, bool from_CCD) :
		vinds({ v0, v1, v2, v3 }), pair_type(EE), point(point)
	{
		using namespace ADU;
		dist = result.distance;
		distSqr = result.sqrDistance;

		if (result.barycentric[2] < -5) {
			// ee
			// C0 = P[0] + s[0] * (P[1] - P[0]) = (1-s[0])*P[0] + s[0]*P[1] for 0 <= s[0] <= 1 
			// C1 = Q[0] + s[1] * (Q[1] - Q[0]) = (1-s[1])*Q[0] + s[1]*Q[1] for 0 <= s[1] <= 1
			// h = C0 - C1
			bary = { 1 - result.barycentric[0], result.barycentric[0],
				result.barycentric[1] - 1, -result.barycentric[1] };
			ini_bary = { 1 - result.barycentric[0], result.barycentric[0], 1 - result.barycentric[1], result.barycentric[1] };

			pair_type == EE;
		}
		else {
			// pt
			bary = { 1.0_r, -result.barycentric[0], -result.barycentric[1], -result.barycentric[2] };
			ini_bary = { 1.0_r, result.barycentric[0], result.barycentric[2], result.barycentric[2] };
		
			pair_type == PT;
		}
		
		Vecf_3 h = result.closest[0] - result.closest[1];
		h_cN = normal.dot(h);

		if (!from_CCD && h_cN < 0) {
			this->normal = -normal;
			h_cN = -h_cN;
		}
		else if (from_CCD && h_cN > 0) {
			this->normal = -normal;
			h_cN = -h_cN;
		}
		else this->normal = normal;

		h_cN -= ContactParameter::get_thickness();
	}

	ContactInfo(const ADU::DistEE<ADU::Real>::Result& result, size_t v0, size_t v1, size_t v2, size_t v3,
		const ADU::Vecf_3& normal, const ADU::Vecf_3& point, bool from_CCD) :
		vinds({ v0, v1, v2, v3 }), pair_type(EE), point(point)
	{
		using namespace ADU;
		dist = result.distance;
		distSqr = result.sqrDistance;

		// C0 = P[0] + s[0] * (P[1] - P[0]) = (1-s[0])*P[0] + s[0]*P[1] for 0 <= s[0] <= 1 
		// C1 = Q[0] + s[1] * (Q[1] - Q[0]) = (1-s[1])*Q[0] + s[1]*Q[1] for 0 <= s[1] <= 1
		// h = C0 - C1
		bary = { 1 - result.parameter[0], result.parameter[0],
			result.parameter[1] - 1, -result.parameter[1] };
		ini_bary = { 1 - result.parameter[0], result.parameter[0], 1 - result.parameter[1], result.parameter[1] };

		Vecf_3 h = result.closest[0] - result.closest[1];
		h_cN = normal.dot(h);

		if (!from_CCD && h_cN < 0) {
			this->normal = -normal;
			h_cN = -h_cN;
		}
		else if (from_CCD && h_cN > 0) {
			this->normal = -normal;
			h_cN = -h_cN;
		}
		else this->normal = normal;

		h_cN -= ContactParameter::get_thickness();

	}

	ContactInfo(const ADU::DistPT<ADU::Real>::Result& result, size_t v0, size_t v1, size_t v2, size_t v3,
		const ADU::Vecf_3& normal, const ADU::Vecf_3& point, bool from_CCD) :
		vinds({ v0, v1, v2, v3 }), pair_type(PT), point(point)
	{
		using namespace ADU;
		dist = result.distance;
		distSqr = result.sqrDistance;

		// C0 = V0
		// C1 = sum_{ i = 0 } ^ 2 b[i] * V[i]
		// where 0 <= b[i] <= 1 for all i and sum_{i=0}^2 b[i] = 1.
		// h = C0 - C1
		bary = { 1.0_r, -result.barycentric[0], -result.barycentric[1], -result.barycentric[2] };
		ini_bary = { 1.0_r, result.barycentric[0], result.barycentric[2], result.barycentric[2] };

		Vecf_3 h = result.closest[0] - result.closest[1];

		h_cN = normal.dot(h);
		if (!from_CCD && h_cN < 0) {
			this->normal = -normal;
			h_cN = -h_cN;
		}
		else if (from_CCD && h_cN > 0) {
			this->normal = -normal;
			h_cN = -h_cN;
		}
		else this->normal = normal;

		h_cN -= ContactParameter::get_thickness();
	}     

};


struct ContactCompare {
	bool operator() (const ContactInfo& c1, const ContactInfo& c2) const {
		if (c1.point.y() != c2.point.y()) return c1.point.y() < c2.point.y();
		if (c1.point.x() != c2.point.x()) return c1.point.x() < c2.point.x();
		if (c1.point.z() != c2.point.z()) return c1.point.z() < c2.point.z();
		return true;
		/*for (int i = 0; i < 4; i++) {
			if (c1.vinds[i] != c2.vinds[i]) return c1.vinds[i] < c2.vinds[i];
		}
		return true;*/
	}
};