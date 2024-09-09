#pragma once

#include "mutils/common_types.h"

class PhyxMaterial {
public:

	ADU::Real density_m2{};
	ADU::Real density_m3{};

	ADU::Real tri_stretch_k{};
	ADU::Real tri_stretch_min{};
	ADU::Real tri_stretch_max{};
	ADU::Real tri_bending_k{};
	bool tri_use_limit{};

	ADU::Real tet_stretch_k{};
	ADU::Real tet_stretch_min{};
	ADU::Real tet_stretch_max{};
	bool tet_use_limit{};


	// triangle
	PhyxMaterial(ADU::Real density_m2, ADU::Real stretch, ADU::Real stretch_min, ADU::Real stretch_max, ADU::Real bending, bool use_limit) {
		this->density_m2 = density_m2;
		this->tri_stretch_k = stretch;
		this->tri_bending_k = bending;
		this->tri_stretch_min = stretch_min;
		this->tri_stretch_max = stretch_max;
		this->tri_use_limit = use_limit;
	}

	// tetrahedral
	PhyxMaterial(ADU::Real density_m3, ADU::Real stretch, ADU::Real stretch_min, ADU::Real stretch_max, bool use_limit) {
		this->density_m3 = density_m3;
		this->tet_stretch_k = stretch;
		this->tet_stretch_min = stretch_min;
		this->tet_stretch_max = stretch_max;
		this->tet_use_limit = use_limit;
	}

};

