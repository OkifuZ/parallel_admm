#include "constraint/XPBD_constraints.h"
#include "mutils/exception_handle.h"
#include "constraint/bending_constraint.h"
#include <mutils/common_type_hostonly.h>


bool FEMTriangleConstraint::initConstraint(const ADU::Matf_X3& verts, const unsigned int particle1, const unsigned int particle2,
	const unsigned int particle3, const ADU::Real xxStiffness, const ADU::Real yyStiffness, const ADU::Real xyStiffness,
	const ADU::Real xyPoissonRatio, const ADU::Real yxPoissonRatio)
{
	m_xxStiffness = xxStiffness;
	m_yyStiffness = yyStiffness;
	m_xyStiffness = xyStiffness;
	m_xyPoissonRatio = xyPoissonRatio;
	m_yxPoissonRatio = yxPoissonRatio;
	m_bodies[0] = particle1;
	m_bodies[1] = particle2;
	m_bodies[2] = particle3;

	const ADU::Vecf_3& x1 = verts.row(particle1);
	const ADU::Vecf_3& x2 = verts.row(particle2);
	const ADU::Vecf_3& x3 = verts.row(particle3);

	return XPBD_utils::init_FEMTriangleConstraint(x1, x2, x3, m_area, m_invRestMat);
}

bool FEMTriangleConstraint::solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt) {
	const unsigned i1 = m_bodies[0];
	const unsigned i2 = m_bodies[1];
	const unsigned i3 = m_bodies[2];

	auto& x1 = verts.row(i1);
	auto& x2 = verts.row(i2);
	auto& x3 = verts.row(i3);

	const ADU::Real invMass1 = inv_mass(i1);
	const ADU::Real invMass2 = inv_mass(i2);
	const ADU::Real invMass3 = inv_mass(i3);

	if (iter == 0)
		m_lambda = 0.0;


	ADU::Vecf_3 corr1, corr2, corr3;
	
	bool in_ADMM = (beta.rows() > 0);
	bool res{};
	if (in_ADMM) {
		 res = XPBD_utils::solve_FEMTriangleConstraint(
			x1, invMass1,
			x2, invMass2,
			x3, invMass3,
			beta.row(i1),
			beta.row(i2),
			beta.row(i3),
			in_ADMM,
			m_area,
			m_invRestMat,
			m_xxStiffness,
			m_yyStiffness,
			m_xyStiffness,
			m_xyPoissonRatio,
			m_yxPoissonRatio,
			stiffness,
			dt,
			m_lambda,
			corr1, corr2, corr3);
	}
	else {
		res = XPBD_utils::solve_FEMTriangleConstraint(
			x1, invMass1,
			x2, invMass2,
			x3, invMass3,
			ADU::Vecf_3{},
			ADU::Vecf_3{},
			ADU::Vecf_3{},
			in_ADMM,
			m_area,
			m_invRestMat,
			m_xxStiffness,
			m_yyStiffness,
			m_xyStiffness,
			m_xyPoissonRatio,
			m_yxPoissonRatio,
			stiffness,
			dt,
			m_lambda,
			corr1, corr2, corr3);
	}



	if (res)
	{
		if (invMass1 != 0.0)
			x1 += corr1;
		if (invMass2 != 0.0)
			x2 += corr2;
		if (invMass3 != 0.0)
			x3 += corr3;
	}
	/*else {
		ADU::make_exception("XPBD solve tri issue");
	}*/
	return res;
}