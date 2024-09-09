#pragma once
#include "mutils/common_types.h"
#include <vector>


struct CoolColor {

	static ADU::Vecf_3 color_converter(int hexValue) {
		ADU::Vecf_3 rgb;
		rgb.x() = ((hexValue >> 16) & 0xFF) / 255.0;  // Extract the RR byte
		rgb.y() = ((hexValue >> 8) & 0xFF) / 255.0;   // Extract the GG byte
		rgb.z() = ((hexValue) & 0xFF) / 255.0;        // Extract the BB byte
		return rgb;
	}

	static ADU::Vecf_3 color_converter(int r, int g, int b) {
		ADU::Vecf_3 rgb;
		rgb.x() = r / 255.0;  // Extract the RR byte
		rgb.y() = g / 255.0;   // Extract the GG byte
		rgb.z() = b / 255.0;        // Extract the BB byte
		return rgb;
	}


	static ADU::Vecf_3 color(int i) {
		return color_converter(good_colors[i % good_colors.size()]);
	}


private:
	static std::vector<int> good_colors;

};