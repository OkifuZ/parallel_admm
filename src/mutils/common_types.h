#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>

constexpr auto ADMM_VERBOSE = false;

namespace ADU {
	using Real = float;

	inline Real operator "" _r(long double value) {
		return static_cast<Real>(value);
	}

	inline Real toReal(long double value) {
		return static_cast<Real>(value);
	}

	constexpr auto DefaultOreder = Eigen::RowMajor;

	using Mati_XX = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
	using Mati_X2 = Eigen::Matrix<int, Eigen::Dynamic, 2, DefaultOreder>;
	using Mati_X3 = Eigen::Matrix<int, Eigen::Dynamic, 3, DefaultOreder>;
	using Mati_X4 = Eigen::Matrix<int, Eigen::Dynamic, 4, DefaultOreder>;

	using Mati_XX_RM = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using Mati_X2_RM = Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>;
	using Mati_X3_RM = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
	using Mati_X4_RM = Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>;

	using Matf_XX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
	using Matf_X3 = Eigen::Matrix<Real, Eigen::Dynamic, 3, DefaultOreder>;
	using Matf_3X = Eigen::Matrix<Real, 3, Eigen::Dynamic>;
	using Matf_X2 = Eigen::Matrix<Real, Eigen::Dynamic, 2, DefaultOreder>;
	using Matf_2X = Eigen::Matrix<Real, 2, Eigen::Dynamic>;

	using Matf_XX_RM = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using Matf_X3_RM = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;
	using Matf_X2_RM = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>;

	using Matf_22 = Eigen::Matrix<Real, 2, 2>;
	using Matf_23 = Eigen::Matrix<Real, 2, 3>;
	using Matf_32 = Eigen::Matrix<Real, 3, 2>;
	using Matf_33 = Eigen::Matrix<Real, 3, 3>;
	using Matf_34 = Eigen::Matrix<Real, 3, 4>;
	using Matf_43 = Eigen::Matrix<Real, 4, 3>;
	using Matf_44 = Eigen::Matrix<Real, 4, 4>;

	using Quatf = Eigen::Quaternion<Real>;

	using Vecf_X = Eigen::Vector<Real, Eigen::Dynamic>;
	using Veci_X = Eigen::Vector<int, Eigen::Dynamic>;

	using Vecf_2 = Eigen::Vector<Real, 2>;
	using Vecf_3 = Eigen::Vector<Real, 3>;
	using Vecf_4 = Eigen::Vector<Real, 4>;
	using Veci_2 = Eigen::Vector<int, 2>;
	using Veci_3 = Eigen::Vector<int, 3>;
	using Veci_4 = Eigen::Vector<int, 4>;


}

