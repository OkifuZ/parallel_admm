#pragma once

#include "mutils/common_types.h"

class ContactParameter {
	static ADU::Real thickness;
	static ADU::Real radius;



public:
	static ADU::Real precision;

	static ADU::Real get_thickness() {
		return thickness;
	}

	static ADU::Real get_broadphase_radius() {
		return radius;
	}

	static void set_thickness(ADU::Real thickness) {
		ContactParameter::thickness = thickness;
	}

	static void set_broadphase_radius(ADU::Real radius) {
		ContactParameter::radius = radius;
	}
};

