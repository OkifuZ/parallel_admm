#include "constraint/xpbd/xpbd_constraint.h"
#include "constraint/bending_constraint.h"




bool XPBD_FEMTetConstraint::initConstraint(const ADU::Matf_X3& verts, const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const unsigned int particle4,
	const ADU::Real stiffness, const ADU::Real poissonRatio) 
{
	using namespace ADU;

	m_stiffness = stiffness;
	m_poissonRatio = poissonRatio;
	m_bodies[0] = particle1;
	m_bodies[1] = particle2;
	m_bodies[2] = particle3;
	m_bodies[3] = particle4;

	const Vecf_3& x1 = verts.row(particle1);
	const Vecf_3& x2 = verts.row(particle2);
	const Vecf_3& x3 = verts.row(particle3);
	const Vecf_3& x4 = verts.row(particle4);

	return XPBD_utils::init_FEMTetraConstraint(x1, x2, x3, x4, m_volume, m_invRestMat);
}


bool XPBD_FEMTetConstraint::solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt) {
	using namespace ADU;

	const unsigned i1 = m_bodies[0];
	const unsigned i2 = m_bodies[1];
	const unsigned i3 = m_bodies[2];
	const unsigned i4 = m_bodies[3];

	auto& x1 = verts.row(i1);
	auto& x2 = verts.row(i2);
	auto& x3 = verts.row(i3);
	auto& x4 = verts.row(i4);

	const Real invMass1 = inv_mass(i1);
	const Real invMass2 = inv_mass(i2);
	const Real invMass3 = inv_mass(i3);
	const Real invMass4 = inv_mass(i4);

	Real currentVolume = -static_cast<Real>(1.0 / 6.0) * (x4 - x1).dot((x3 - x1).cross(x2 - x1));
	bool handleInversion = false;
	//if (currentVolume / m_volume < 0.2)		// Only 20% of initial volume left
	//	handleInversion = true;

	if (iter == 0)
		m_lambda = 0.0;

	bool in_ADMM = (beta.rows() > 0);
	bool res{};
	
	Vecf_3 corr1, corr2, corr3, corr4;
	
	if (in_ADMM) {
		res = XPBD_utils::solve_FEMTetraConstraint(
			x1, invMass1,
			x2, invMass2,
			x3, invMass3,
			x4, invMass4,
			beta.row(i1),
			beta.row(i2),
			beta.row(i3),
			beta.row(i4),
			in_ADMM,
			m_volume,
			m_invRestMat,
			m_stiffness,
			m_poissonRatio, handleInversion,
			dt,
			m_lambda,
			corr1, corr2, corr3, corr4);
	}
	else {
		res = XPBD_utils::solve_FEMTetraConstraint(
			x1, invMass1,
			x2, invMass2,
			x3, invMass3,
			x4, invMass4,
			Vecf_3{},
			Vecf_3{},
			Vecf_3{},
			Vecf_3{},
			in_ADMM,
			m_volume,
			m_invRestMat,
			m_stiffness,
			m_poissonRatio, handleInversion,
			dt,
			m_lambda,
			corr1, corr2, corr3, corr4);
	}
	
	if (res)
	{
		if (invMass1 != 0.0)
			x1 += corr1;
		if (invMass2 != 0.0)
			x2 += corr2;
		if (invMass3 != 0.0)
			x3 += corr3;
		if (invMass4 != 0.0)
			x4 += corr4;
	}
	return res;
}
