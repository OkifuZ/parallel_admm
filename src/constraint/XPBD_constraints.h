#pragma once
#include "mutils/common_types.h"
//#include "constraint/bending_constraint.h"

class Constraint;

class XPBDConstraint
{
public:
	/** indices of the linked bodies */
	std::vector<unsigned int> m_bodies;
	int type = 0; // 0 tri, 1 bending, 2 tet

	ADU::Real m_lambda;

	ADU::Real stiffness = 1.0;

	XPBDConstraint(const unsigned int numberOfBodies)
	{
		m_bodies.resize(numberOfBodies);
	}

	unsigned int numberOfBodies() const { return static_cast<unsigned int>(m_bodies.size()); }
	virtual ~XPBDConstraint() {};

	virtual bool initConstraintBeforeProjection() { return true; };
	virtual bool updateConstraint() { return true; };
	virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta,
		const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt) { return true; };
	/*virtual bool solvePositionConstraint(ADU::Matf_X3& verts,
		const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt) {
		return true;
	};*/
	virtual bool solveVelocityConstraint(const unsigned int iter) { return true; };
};

class FEMTriangleConstraint : public XPBDConstraint
{
public:
	ADU::Real m_area;
	ADU::Matf_22 m_invRestMat;
	ADU::Real m_xxStiffness;
	ADU::Real m_xyStiffness;
	ADU::Real m_yyStiffness;
	ADU::Real m_xyPoissonRatio;
	ADU::Real m_yxPoissonRatio;

	FEMTriangleConstraint() : XPBDConstraint(3) { type = 0; }

	virtual bool initConstraint(const ADU::Matf_X3& verts, const unsigned int particle1, const unsigned int particle2,
		const unsigned int particle3, const ADU::Real xxStiffness, const ADU::Real yyStiffness, const ADU::Real xyStiffness,
		const ADU::Real xyPoissonRatio, const ADU::Real yxPoissonRatio);
	virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
	//virtual bool solvePositionConstraint(ADU::Matf_X3& verts, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
};

class XPBDBendingConstraint : public XPBDConstraint
{
public:

	ADU::Vecf_3 rest_mean_curvature{};
	ADU::Real rest_mean_curvature_norm{};
	std::vector<ADU::Real> coeffs; // middle vert, ring verts

	XPBDBendingConstraint() : XPBDConstraint(0) { type = 1; }

	virtual bool initConstraint(const std::shared_ptr<Constraint>& bc);
	virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
	//virtual bool solvePositionConstraint(ADU::Matf_X3& verts, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
};

class XPBD_FEMTetConstraint : public XPBDConstraint
{
public:
	ADU::Real m_stiffness;
	ADU::Real m_poissonRatio;
	ADU::Real m_volume;
	ADU::Matf_33 m_invRestMat;


	XPBD_FEMTetConstraint() : XPBDConstraint(4) { type = 2; }

	virtual bool initConstraint(const ADU::Matf_X3& verts, const unsigned int particle1, const unsigned int particle2,
		const unsigned int particle3, const unsigned int particle4,
		const ADU::Real stiffness, const ADU::Real poissonRatio);
	virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
	//virtual bool solvePositionConstraint(ADU::Matf_X3& verts, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
};


namespace XPBD_utils {
	static ADU::Real eps = static_cast<ADU::Real>(1e-9);

	static bool init_FEMTriangleConstraint(
		const ADU::Vecf_3& p0,
		const ADU::Vecf_3& p1,
		const ADU::Vecf_3& p2,
		ADU::Real& area,
		ADU::Matf_22& invRestMat)
	{
		using namespace ADU;
		Vecf_3 normal0 = (p1 - p0).cross(p2 - p0);
		area = normal0.norm() * static_cast<Real>(0.5);

		Vecf_3 axis0_1 = p1 - p0;
		axis0_1.normalize();
		Vecf_3 axis0_2 = normal0.cross(axis0_1);
		axis0_2.normalize();

		Vecf_2 p[3];
		p[0] = Vecf_2(p0.dot(axis0_2), p0.dot(axis0_1));
		p[1] = Vecf_2(p1.dot(axis0_2), p1.dot(axis0_1));
		p[2] = Vecf_2(p2.dot(axis0_2), p2.dot(axis0_1));

		Matf_22 P;
		P(0, 0) = p[0][0] - p[2][0];
		P(1, 0) = p[0][1] - p[2][1];
		P(0, 1) = p[1][0] - p[2][0];
		P(1, 1) = p[1][1] - p[2][1];

		const Real det = P.determinant();
		if (fabs(det) > eps)
		{
			invRestMat = P.inverse();
			return true;
		}
		return false;
		//Vecf_3 e12 = p1 - p0;
		//Vecf_3 e13 = p2 - p0;
		//Vecf_3 n1 = e12.normalized();
		//Vecf_3 n2 = (e13 - e13.dot(n1) * n1).normalized();
		//Matf_32 basis;
		//Matf_32 edges;
		//basis.col(0) = n1; basis.col(1) = n2;
		//edges.col(0) = e12; edges.col(1) = e13;
		//invRestMat = (basis.transpose() * edges).inverse(); // Rest pose matrix

		//area = (basis.transpose() * edges).determinant() * 0.5_r;

		//if (area < 0) {
		//	return false;
		//}
		//return true;

	}


	static bool solve_FEMTriangleConstraint(
		const ADU::Vecf_3& p0, ADU::Real invMass0,
		const ADU::Vecf_3& p1, ADU::Real invMass1,
		const ADU::Vecf_3& p2, ADU::Real invMass2,
		const ADU::Vecf_3& beta0,
		const ADU::Vecf_3& beta1,
		const ADU::Vecf_3& beta2,
		bool in_ADMM,
		const ADU::Real& area,
		const ADU::Matf_22& invRestMat,
		const ADU::Real youngsModulusX,
		const ADU::Real youngsModulusY,
		const ADU::Real youngsModulusShear,
		const ADU::Real poissonRatioXY,
		const ADU::Real poissonRatioYX,
		const ADU::Real stiffness,
		const ADU::Real dt,
		ADU::Real& multiplier,
		ADU::Vecf_3& corr0, ADU::Vecf_3& corr1, ADU::Vecf_3& corr2)
	{
		using namespace ADU;
		// Orthotropic elasticity tensor
		Matf_33 C;
		C.setZero();
		C(0, 0) = youngsModulusX / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(0, 1) = youngsModulusX * poissonRatioYX / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(1, 1) = youngsModulusY / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(1, 0) = youngsModulusY * poissonRatioXY / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(2, 2) = youngsModulusShear;

		// Determine \partial x/\partial m_i
		Eigen::Matrix<Real, 3, 2> F;
		const ADU::Vecf_3 p13 = p0 - p2;
		const ADU::Vecf_3 p23 = p1 - p2;
		F(0, 0) = p13[0] * invRestMat(0, 0) + p23[0] * invRestMat(1, 0);
		F(0, 1) = p13[0] * invRestMat(0, 1) + p23[0] * invRestMat(1, 1);
		F(1, 0) = p13[1] * invRestMat(0, 0) + p23[1] * invRestMat(1, 0);
		F(1, 1) = p13[1] * invRestMat(0, 1) + p23[1] * invRestMat(1, 1);
		F(2, 0) = p13[2] * invRestMat(0, 0) + p23[2] * invRestMat(1, 0);
		F(2, 1) = p13[2] * invRestMat(0, 1) + p23[2] * invRestMat(1, 1);

		// epsilon = 0.5(F^T * F - I)
		Matf_22 epsilon;
		epsilon(0, 0) = static_cast<Real>(0.5) * (F(0, 0) * F(0, 0) + F(1, 0) * F(1, 0) + F(2, 0) * F(2, 0) - static_cast<Real>(1.0));		// xx
		epsilon(1, 1) = static_cast<Real>(0.5) * (F(0, 1) * F(0, 1) + F(1, 1) * F(1, 1) + F(2, 1) * F(2, 1) - static_cast<Real>(1.0));		// yy
		epsilon(0, 1) = static_cast<Real>(0.5) * (F(0, 0) * F(0, 1) + F(1, 0) * F(1, 1) + F(2, 0) * F(2, 1));			// xy
		epsilon(1, 0) = epsilon(0, 1);

		// P(F) = det(F) * C*E * F^-T => E = green strain
		Matf_22 stress;
		stress(0, 0) = C(0, 0) * epsilon(0, 0) + C(0, 1) * epsilon(1, 1) + C(0, 2) * epsilon(0, 1);
		stress(1, 1) = C(1, 0) * epsilon(0, 0) + C(1, 1) * epsilon(1, 1) + C(1, 2) * epsilon(0, 1);
		stress(0, 1) = C(2, 0) * epsilon(0, 0) + C(2, 1) * epsilon(1, 1) + C(2, 2) * epsilon(0, 1);
		stress(1, 0) = stress(0, 1);

		const Eigen::Matrix<Real, 3, 2> piolaKirchhoffStres = F * stress;

		Real psi = 0.0;
		for (unsigned char j = 0; j < 2; j++)
			for (unsigned char k = 0; k < 2; k++)
				psi += epsilon(j, k) * stress(j, k);
		psi = static_cast<Real>(0.5) * psi;
		Real energy = area * psi;

		// compute gradient
		Eigen::Matrix<Real, 3, 2> H = area * piolaKirchhoffStres * invRestMat.transpose();
		Vecf_3 gradC[3];
		for (unsigned char j = 0; j < 3; ++j)
		{
			gradC[0][j] = H(j, 0);
			gradC[1][j] = H(j, 1);
		}
		gradC[2] = -gradC[0] - gradC[1];


		Real sum_normGradC = invMass0 * gradC[0].squaredNorm();
		sum_normGradC += invMass1 * gradC[1].squaredNorm();
		sum_normGradC += invMass2 * gradC[2].squaredNorm();

		Real grad_c_beta = 0.0_r;
		if (in_ADMM) {
			grad_c_beta += gradC[0].dot(beta0);
			grad_c_beta += gradC[1].dot(beta1);
			grad_c_beta += gradC[2].dot(beta2);
		}

		// exit early if required
		if (fabs(sum_normGradC) > eps)
		{
			Real alpha = 1.0_r / ((youngsModulusX + youngsModulusY + youngsModulusShear) * 0.33 * dt * dt);
			// compute scaling factor
			//const Real s = energy / (sum_normGradC + alpha);

			//const Real s_0 = (energy + alpha * multiplier) / (sum_normGradC + alpha);
			const Real s = (energy + alpha * multiplier - grad_c_beta) / (sum_normGradC + alpha);
			//const Real s = (energy + alpha * multiplier) / (sum_normGradC + alpha);
			multiplier -= s;

			corr0 = -(s * invMass0) * gradC[0];
			corr1 = -(s * invMass1) * gradC[1];
			corr2 = -(s * invMass2) * gradC[2];

			//// update positions
			//if (in_ADMM) {
			//	corr0 = -(s * invMass0) * gradC[0] - beta0;
			//	corr1 = -(s * invMass1) * gradC[1] - beta1;
			//	corr2 = -(s * invMass2) * gradC[2] - beta2;
			//}
			//else {
			//	corr0 = -(s * invMass0) * gradC[0];
			//	corr1 = -(s * invMass1) * gradC[1];
			//	corr2 = -(s * invMass2) * gradC[2];
			//}

			return true;
		}
		/*else {
			if (in_ADMM) {
				corr0 = - beta0;
				corr1 = - beta1;
				corr2 = - beta2;
				return true;
			}
		}*/

		return false;
	}

	static bool get_FEMTriangleGradient(
		const ADU::Vecf_3& p0,
		const ADU::Vecf_3& p1,
		const ADU::Vecf_3& p2,
		const ADU::Real& area,
		const ADU::Matf_22& invRestMat,
		const ADU::Real youngsModulusX,
		const ADU::Real youngsModulusY,
		const ADU::Real youngsModulusShear,
		const ADU::Real poissonRatioXY,
		const ADU::Real poissonRatioYX,
		std::vector<ADU::Vecf_3>& gradient,
		ADU::Real& c)
	{
		using namespace ADU;
		// Orthotropic elasticity tensor
		Matf_33 C;
		C.setZero();
		C(0, 0) = youngsModulusX / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(0, 1) = youngsModulusX * poissonRatioYX / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(1, 1) = youngsModulusY / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(1, 0) = youngsModulusY * poissonRatioXY / (static_cast<Real>(1.0) - poissonRatioXY * poissonRatioYX);
		C(2, 2) = youngsModulusShear;

		// Determine \partial x/\partial m_i
		Eigen::Matrix<Real, 3, 2> F;
		const ADU::Vecf_3 p13 = p0 - p2;
		const ADU::Vecf_3 p23 = p1 - p2;
		F(0, 0) = p13[0] * invRestMat(0, 0) + p23[0] * invRestMat(1, 0);
		F(0, 1) = p13[0] * invRestMat(0, 1) + p23[0] * invRestMat(1, 1);
		F(1, 0) = p13[1] * invRestMat(0, 0) + p23[1] * invRestMat(1, 0);
		F(1, 1) = p13[1] * invRestMat(0, 1) + p23[1] * invRestMat(1, 1);
		F(2, 0) = p13[2] * invRestMat(0, 0) + p23[2] * invRestMat(1, 0);
		F(2, 1) = p13[2] * invRestMat(0, 1) + p23[2] * invRestMat(1, 1);

		// epsilon = 0.5(F^T * F - I)
		Matf_22 epsilon;
		epsilon(0, 0) = static_cast<Real>(0.5) * (F(0, 0) * F(0, 0) + F(1, 0) * F(1, 0) + F(2, 0) * F(2, 0) - static_cast<Real>(1.0));		// xx
		epsilon(1, 1) = static_cast<Real>(0.5) * (F(0, 1) * F(0, 1) + F(1, 1) * F(1, 1) + F(2, 1) * F(2, 1) - static_cast<Real>(1.0));		// yy
		epsilon(0, 1) = static_cast<Real>(0.5) * (F(0, 0) * F(0, 1) + F(1, 0) * F(1, 1) + F(2, 0) * F(2, 1));			// xy
		epsilon(1, 0) = epsilon(0, 1);

		// P(F) = det(F) * C*E * F^-T => E = green strain
		Matf_22 stress;
		stress(0, 0) = C(0, 0) * epsilon(0, 0) + C(0, 1) * epsilon(1, 1) + C(0, 2) * epsilon(0, 1);
		stress(1, 1) = C(1, 0) * epsilon(0, 0) + C(1, 1) * epsilon(1, 1) + C(1, 2) * epsilon(0, 1);
		stress(0, 1) = C(2, 0) * epsilon(0, 0) + C(2, 1) * epsilon(1, 1) + C(2, 2) * epsilon(0, 1);
		stress(1, 0) = stress(0, 1);

		const Eigen::Matrix<Real, 3, 2> piolaKirchhoffStres = F * stress;

		Real psi = 0.0;
		for (unsigned char j = 0; j < 2; j++)
			for (unsigned char k = 0; k < 2; k++)
				psi += epsilon(j, k) * stress(j, k);
		psi = static_cast<Real>(0.5) * psi;
		Real energy = area * psi;

		c = energy;

		// compute gradient
		Eigen::Matrix<Real, 3, 2> H = area * piolaKirchhoffStres * invRestMat.transpose();
		Vecf_3 gradC[3];
		for (unsigned char j = 0; j < 3; ++j)
		{
			gradC[0][j] = H(j, 0);
			gradC[1][j] = H(j, 1);
		}
		gradC[2] = -gradC[0] - gradC[1];
		gradient.resize(3);
		gradient[0] = gradC[0];
		gradient[1] = gradC[1];
		gradient[2] = gradC[2];

		return true;
	}

	static bool init_FEMTetraConstraint(			// compute only when rest shape changes
		const ADU::Vecf_3& p0,
		const ADU::Vecf_3& p1,
		const ADU::Vecf_3& p2,
		const ADU::Vecf_3& p3,
		ADU::Real& volume,
		ADU::Matf_33& invRestMat)
	{
		using namespace ADU;
		volume = fabs(static_cast<Real>(1.0 / 6.0) * (p3 - p0).dot((p2 - p0).cross(p1 - p0)));

		Matf_33 m;
		m.col(0) = p0 - p3;
		m.col(1) = p1 - p3;
		m.col(2) = p2 - p3;

		Real det = m.determinant();
		if (fabs(det) > eps)
		{
			invRestMat = m.inverse();
			return true;
		}
		return false;
	}

	static void computeGreenStrainAndPiolaStress(
		const ADU::Vecf_3& x1, const ADU::Vecf_3& x2, const ADU::Vecf_3& x3, const ADU::Vecf_3& x4,
		const ADU::Matf_33& invRestMat,
		const ADU::Real restVolume,
		const ADU::Real mu, const ADU::Real lambda, ADU::Matf_33& epsilon, ADU::Matf_33& sigma, ADU::Real& energy)
	{
		// Determine \partial x/\partial m_i
		ADU::Matf_33 F;
		const ADU::Vecf_3 p14 = x1 - x4;
		const ADU::Vecf_3 p24 = x2 - x4;
		const ADU::Vecf_3 p34 = x3 - x4;
		F(0, 0) = p14[0] * invRestMat(0, 0) + p24[0] * invRestMat(1, 0) + p34[0] * invRestMat(2, 0);
		F(0, 1) = p14[0] * invRestMat(0, 1) + p24[0] * invRestMat(1, 1) + p34[0] * invRestMat(2, 1);
		F(0, 2) = p14[0] * invRestMat(0, 2) + p24[0] * invRestMat(1, 2) + p34[0] * invRestMat(2, 2);

		F(1, 0) = p14[1] * invRestMat(0, 0) + p24[1] * invRestMat(1, 0) + p34[1] * invRestMat(2, 0);
		F(1, 1) = p14[1] * invRestMat(0, 1) + p24[1] * invRestMat(1, 1) + p34[1] * invRestMat(2, 1);
		F(1, 2) = p14[1] * invRestMat(0, 2) + p24[1] * invRestMat(1, 2) + p34[1] * invRestMat(2, 2);

		F(2, 0) = p14[2] * invRestMat(0, 0) + p24[2] * invRestMat(1, 0) + p34[2] * invRestMat(2, 0);
		F(2, 1) = p14[2] * invRestMat(0, 1) + p24[2] * invRestMat(1, 1) + p34[2] * invRestMat(2, 1);
		F(2, 2) = p14[2] * invRestMat(0, 2) + p24[2] * invRestMat(1, 2) + p34[2] * invRestMat(2, 2);

		// epsilon = 1/2 F^T F - I

		epsilon(0, 0) = static_cast<ADU::Real>(0.5) * (F(0, 0) * F(0, 0) + F(1, 0) * F(1, 0) + F(2, 0) * F(2, 0) - static_cast<ADU::Real>(1.0));		// xx
		epsilon(1, 1) = static_cast<ADU::Real>(0.5) * (F(0, 1) * F(0, 1) + F(1, 1) * F(1, 1) + F(2, 1) * F(2, 1) - static_cast<ADU::Real>(1.0));		// yy
		epsilon(2, 2) = static_cast<ADU::Real>(0.5) * (F(0, 2) * F(0, 2) + F(1, 2) * F(1, 2) + F(2, 2) * F(2, 2) - static_cast<ADU::Real>(1.0));		// zz
		epsilon(0, 1) = static_cast<ADU::Real>(0.5) * (F(0, 0) * F(0, 1) + F(1, 0) * F(1, 1) + F(2, 0) * F(2, 1));			// xy
		epsilon(0, 2) = static_cast<ADU::Real>(0.5) * (F(0, 0) * F(0, 2) + F(1, 0) * F(1, 2) + F(2, 0) * F(2, 2));			// xz
		epsilon(1, 2) = static_cast<ADU::Real>(0.5) * (F(0, 1) * F(0, 2) + F(1, 1) * F(1, 2) + F(2, 1) * F(2, 2));			// yz
		epsilon(1, 0) = epsilon(0, 1);
		epsilon(2, 0) = epsilon(0, 2);
		epsilon(2, 1) = epsilon(1, 2);

		// P(F) = F(2 mu E + lambda tr(E)I) => E = green strain
		const ADU::Real trace = epsilon(0, 0) + epsilon(1, 1) + epsilon(2, 2);
		const ADU::Real ltrace = lambda * trace;
		sigma = epsilon * 2.0 * mu;
		sigma(0, 0) += ltrace;
		sigma(1, 1) += ltrace;
		sigma(2, 2) += ltrace;
		sigma = F * sigma;

		ADU::Real psi = 0.0;
		for (unsigned char j = 0; j < 3; j++)
			for (unsigned char k = 0; k < 3; k++)
				psi += epsilon(j, k) * epsilon(j, k);
		psi = mu * psi + static_cast<ADU::Real>(0.5) * lambda * trace * trace;
		energy = restVolume * psi;
	}

	// ----------------------------------------------------------------------------------------------
	static void computeGradCGreen(ADU::Real restVolume, const ADU::Matf_33& invRestMat, const ADU::Matf_33& sigma, ADU::Vecf_3* J)
	{
		using namespace ADU;
		Matf_33 H;
		Matf_33 T;
		T = invRestMat.transpose();
		H = sigma * T * restVolume;

		J[0][0] = H(0, 0);
		J[1][0] = H(0, 1);
		J[2][0] = H(0, 2);

		J[0][1] = H(1, 0);
		J[1][1] = H(1, 1);
		J[2][1] = H(1, 2);

		J[0][2] = H(2, 0);
		J[1][2] = H(2, 1);
		J[2][2] = H(2, 2);

		J[3] = -J[0] - J[1] - J[2];
	}

	
	static bool solve_FEMTetraConstraint(
		const ADU::Vecf_3& p0, ADU::Real invMass0,
		const ADU::Vecf_3& p1, ADU::Real invMass1,
		const ADU::Vecf_3& p2, ADU::Real invMass2,
		const ADU::Vecf_3& p3, ADU::Real invMass3,
		const ADU::Vecf_3& beta0,
		const ADU::Vecf_3& beta1,
		const ADU::Vecf_3& beta2,
		const ADU::Vecf_3& beta3,
		bool in_ADMM,
		const ADU::Real restVolume,
		const ADU::Matf_33& invRestMat,
		const ADU::Real youngsModulus,
		const ADU::Real poissonRatio,
		const bool  handleInversion,
		const ADU::Real dt,
		ADU::Real& multiplier,
		ADU::Vecf_3& corr0, ADU::Vecf_3& corr1, ADU::Vecf_3& corr2, ADU::Vecf_3& corr3)
	{
		using namespace ADU;
		corr0.setZero();
		corr1.setZero();
		corr2.setZero();
		corr3.setZero();

		if (youngsModulus <= 0.0)
			return true;

		if (poissonRatio < 0.0 || poissonRatio > 0.49)
			return false;


		Vecf_3 gradU_[4];
		Matf_33 epsilon, sigma;
		Real volume = (p1 - p0).cross(p2 - p0).dot(p3 - p0) / static_cast<Real>(6.0);

		// compute the Lame coefficients mu and lambda divided by Young's modulus E
		// since we use Young's modulus as compliance factor alpha = 1/E.
		/*Real mu_ = 1.0 / static_cast<Real>(2.0) / (static_cast<Real>(1.0) + poissonRatio);
		Real lambda_ = 1.0 * poissonRatio / (static_cast<Real>(1.0) + poissonRatio) / (static_cast<Real>(1.0) - static_cast<Real>(2.0) * poissonRatio);*/

		Real mu_ = youngsModulus / static_cast<Real>(2.0) / (static_cast<Real>(1.0) + poissonRatio);
		Real lambda_ = youngsModulus * poissonRatio / (static_cast<Real>(1.0) + poissonRatio) / (static_cast<Real>(1.0) - static_cast<Real>(2.0) * poissonRatio);

		// compute value U' which is the potential energy of the elastic solid U divided by Young's modulus E
		// U = E * U'
		Real U_ = 0.0;
		if (!handleInversion || volume > 0.0)
		{
			computeGreenStrainAndPiolaStress(p0, p1, p2, p3, invRestMat, restVolume, mu_, lambda_, epsilon, sigma, U_);
			computeGradCGreen(restVolume, invRestMat, sigma, gradU_);
		}
		/*else
		{
			computeGreenStrainAndPiolaStressInversion(p0, p1, p2, p3, invRestMat, restVolume, mu_, lambda_, epsilon, sigma, U_);
			computeGradCGreen(restVolume, invRestMat, sigma, gradU_);
		}*/

		// By choosing the constraint function as sqrt(2 U'), the potential energy used in XPBD: 
		// U = 0.5 * alpha^-1 * C^2
		// gives us exactly the required potential energy of the elastic solid. 
		const Real C = sqrt(2.0 * U_);

		Real sum_normGradU_ =
			invMass0 * gradU_[0].squaredNorm() +
			invMass1 * gradU_[1].squaredNorm() +
			invMass2 * gradU_[2].squaredNorm() +
			invMass3 * gradU_[3].squaredNorm();

		Real grad_c_beta = 0.0_r;
		if (in_ADMM) {
			grad_c_beta += gradU_[0].dot(beta0);
			grad_c_beta += gradU_[1].dot(beta1);
			grad_c_beta += gradU_[2].dot(beta2);
			grad_c_beta += gradU_[3].dot(beta2);
		}

		Real alpha = static_cast<Real>(1.0) / (youngsModulus * dt * dt);
		//Real alpha = static_cast<Real>(1.0) / (alpha);
		// Note that grad C = 1/C grad U'
		sum_normGradU_ += C * C * alpha;

		if (sum_normGradU_ < eps)
			return false;

		// compute scaling factor
		//const Real s = (C * C + C * alpha * multiplier) / sum_normGradU_;
		// compute scaling factor
		const Real s = (C * C * C + C * C * alpha * multiplier - grad_c_beta * C) / sum_normGradU_;
		multiplier -= s;

		corr0 = -s * invMass0 * gradU_[0];
		corr1 = -s * invMass1 * gradU_[1];
		corr2 = -s * invMass2 * gradU_[2];
		corr3 = -s * invMass3 * gradU_[3];

		return true;
	}


	// ----------------------------------------------------------------------------------------------
	static bool solve_TrianglePointDistanceConstraint(
		const ADU::Vecf_3& p, ADU::Real invMass,
		const ADU::Vecf_3& p0, ADU::Real invMass0,
		const ADU::Vecf_3& p1, ADU::Real invMass1,
		const ADU::Vecf_3& p2, ADU::Real invMass2,
		const ADU::Real restDist,
		const ADU::Real compressionStiffness,
		const ADU::Real stretchStiffness,
		ADU::Vecf_3& corr, ADU::Vecf_3& corr0, ADU::Vecf_3& corr1, ADU::Vecf_3& corr2)
	{
		using namespace ADU;
		// find barycentric coordinates of closest point on triangle

		ADU::Real b0 = static_cast<ADU::Real>(1.0 / 3.0);		// for singular case
		ADU::Real b1 = b0;
		ADU::Real b2 = b0;

		ADU::Vecf_3 d1 = p1 - p0;
		ADU::Vecf_3 d2 = p2 - p0;
		ADU::Vecf_3 pp0 = p - p0;
		ADU::Real a = d1.dot(d1);
		ADU::Real b = d2.dot(d1);
		ADU::Real c = pp0.dot(d1);
		ADU::Real d = b;
		ADU::Real e = d2.dot(d2);
		ADU::Real f = pp0.dot(d2);
		ADU::Real det = a * e - b * d;

		if (det != 0.0) {
			ADU::Real s = (c * e - b * f) / det;
			ADU::Real t = (a * f - c * d) / det;
			b0 = static_cast<ADU::Real>(1.0) - s - t;		// inside triangle
			b1 = s;
			b2 = t;
			if (b0 < 0.0) {		// on edge 1-2
				ADU::Vecf_3 d = p2 - p1;
				ADU::Real d2 = d.dot(d);
				ADU::Real t = (d2 == static_cast<ADU::Real>(0.0)) ? static_cast<ADU::Real>(0.5) : d.dot(p - p1) / d2;
				if (t < 0.0) t = 0.0;	// on point 1
				if (t > 1.0) t = 1.0;	// on point 2
				b0 = 0.0;
				b1 = (static_cast<ADU::Real>(1.0) - t);
				b2 = t;
			}
			else if (b1 < 0.0) {	// on edge 2-0
				ADU::Vecf_3 d = p0 - p2;
				ADU::Real d2 = d.dot(d);
				ADU::Real t = (d2 == static_cast<ADU::Real>(0.0)) ? static_cast<ADU::Real>(0.5) : d.dot(p - p2) / d2;
				if (t < 0.0) t = 0.0;	// on point 2
				if (t > 1.0) t = 1.0; // on point 0
				b1 = 0.0;
				b2 = (static_cast<ADU::Real>(1.0) - t);
				b0 = t;
			}
			else if (b2 < 0.0) {	// on edge 0-1
				ADU::Vecf_3 d = p1 - p0;
				ADU::Real d2 = d.dot(d);
				ADU::Real t = (d2 == static_cast<ADU::Real>(0.0)) ? static_cast<ADU::Real>(0.5) : d.dot(p - p0) / d2;
				if (t < 0.0) t = 0.0;	// on point 0
				if (t > 1.0) t = 1.0;	// on point 1
				b2 = 0.0;
				b0 = (static_cast<ADU::Real>(1.0) - t);
				b1 = t;
			}
		}
		ADU::Vecf_3 q = p0 * b0 + p1 * b1 + p2 * b2;
		ADU::Vecf_3 n = p - q;
		ADU::Real dist = n.norm();
		n.normalize();
		ADU::Real C = dist - restDist;
		ADU::Vecf_3 grad = n;
		ADU::Vecf_3 grad0 = -n * b0;
		ADU::Vecf_3 grad1 = -n * b1;
		ADU::Vecf_3 grad2 = -n * b2;

		ADU::Real s = invMass + invMass0 * b0 * b0 + invMass1 * b1 * b1 + invMass2 * b2 * b2;
		if (s == 0.0)
			return false;

		s = C / s;
		if (C < 0.0)
			s *= compressionStiffness;
		else
			s *= stretchStiffness;

		if (s == 0.0)
			return false;

		corr = -s * invMass * grad;
		corr0 = -s * invMass0 * grad0;
		corr1 = -s * invMass1 * grad1;
		corr2 = -s * invMass2 * grad2;
		return true;
	}

	// ----------------------------------------------------------------------------------------------
	static bool solve_EdgeEdgeDistanceConstraint(
		const ADU::Vecf_3& p0, ADU::Real invMass0,
		const ADU::Vecf_3& p1, ADU::Real invMass1,
		const ADU::Vecf_3& p2, ADU::Real invMass2,
		const ADU::Vecf_3& p3, ADU::Real invMass3,
		const ADU::Real restDist,
		const ADU::Real compressionStiffness,
		const ADU::Real stretchStiffness,
		ADU::Vecf_3& corr0, ADU::Vecf_3& corr1, ADU::Vecf_3& corr2, ADU::Vecf_3& corr3)
	{
		using namespace ADU;
		ADU::Vecf_3 d0 = p1 - p0;
		ADU::Vecf_3 d1 = p3 - p2;

		ADU::Real a = d0.squaredNorm();
		ADU::Real b = -d0.dot(d1);
		ADU::Real c = d0.dot(d1);
		ADU::Real d = -d1.squaredNorm();
		ADU::Real e = (p2 - p0).dot(d0);
		ADU::Real f = (p2 - p0).dot(d1);
		ADU::Real det = a * d - b * c;
		ADU::Real s, t;
		if (det != 0.0) {
			det = static_cast<ADU::Real>(1.0) / det;
			s = (e * d - b * f) * det;
			t = (a * f - e * c) * det;
		}
		else {	// d0 and d1 parallel
			ADU::Real s0 = p0.dot(d0);
			ADU::Real s1 = p1.dot(d0);
			ADU::Real t0 = p2.dot(d0);
			ADU::Real t1 = p3.dot(d0);
			bool flip0 = false;
			bool flip1 = false;

			if (s0 > s1) { ADU::Real f = s0; s0 = s1; s1 = f; flip0 = true; }
			if (t0 > t1) { ADU::Real f = t0; t0 = t1; t1 = f; flip1 = true; }

			if (s0 >= t1) {
				s = !flip0 ? static_cast<ADU::Real>(0.0) : static_cast<ADU::Real>(1.0);
				t = !flip1 ? static_cast<ADU::Real>(1.0) : static_cast<ADU::Real>(0.0);
			}
			else if (t0 >= s1) {
				s = !flip0 ? static_cast<ADU::Real>(1.0) : static_cast<ADU::Real>(0.0);
				t = !flip1 ? static_cast<ADU::Real>(0.0) : static_cast<ADU::Real>(1.0);
			}
			else {		// overlap
				ADU::Real mid = (s0 > t0) ? (s0 + t1) * static_cast<ADU::Real>(0.5) : (t0 + s1) * static_cast<ADU::Real>(0.5);
				s = (s0 == s1) ? static_cast<ADU::Real>(0.5) : (mid - s0) / (s1 - s0);
				t = (t0 == t1) ? static_cast<ADU::Real>(0.5) : (mid - t0) / (t1 - t0);
			}
		}
		if (s < 0.0) s = 0.0;
		if (s > 1.0) s = 1.0;
		if (t < 0.0) t = 0.0;
		if (t > 1.0) t = 1.0;

		ADU::Real b0 = static_cast<ADU::Real>(1.0) - s;
		ADU::Real b1 = s;
		ADU::Real b2 = static_cast<ADU::Real>(1.0) - t;
		ADU::Real b3 = t;

		ADU::Vecf_3 q0 = p0 * b0 + p1 * b1;
		ADU::Vecf_3 q1 = p2 * b2 + p3 * b3;
		ADU::Vecf_3 n = q0 - q1;
		ADU::Real dist = n.norm();
		n.normalize();
		ADU::Real C = dist - restDist;
		ADU::Vecf_3 grad0 = n * b0;
		ADU::Vecf_3 grad1 = n * b1;
		ADU::Vecf_3 grad2 = -n * b2;
		ADU::Vecf_3 grad3 = -n * b3;

		s = invMass0 * b0 * b0 + invMass1 * b1 * b1 + invMass2 * b2 * b2 + invMass3 * b3 * b3;
		if (s == 0.0)
			return false;

		s = C / s;
		if (C < 0.0)
			s *= compressionStiffness;
		else
			s *= stretchStiffness;

		if (s == 0.0)
			return false;

		corr0 = -s * invMass0 * grad0;
		corr1 = -s * invMass1 * grad1;
		corr2 = -s * invMass2 * grad2;
		corr3 = -s * invMass3 * grad3;
		return true;
	}


}