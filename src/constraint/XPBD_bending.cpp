#include "constraint/XPBD_constraints.h"
#include "constraint/bending_constraint.h"
#include "constraint/bending_constraint.h"



bool XPBDBendingConstraint::initConstraint(const std::shared_ptr<Constraint>& bc_) 
{
	std::shared_ptr<BendingConstraint> bc;
	if (bc_->type == 3) {
		bc = std::static_pointer_cast<BendingConstraint>(bc_);
	}
	else {
		return false;
	}

	this->m_bodies.resize(bc->inds.size());
	for (int i = 0; i < this->m_bodies.size(); i++ ) {
		this->m_bodies[i] = bc->inds[i];
	}

	this->coeffs = bc->coeffs;
	this->rest_mean_curvature = bc->rest_mean_curvature;
	this->rest_mean_curvature_norm = bc->rest_mean_curvature_norm;
	this->stiffness = bc->k;
	return true;
}

bool XPBDBendingConstraint::solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt) {

	using namespace ADU;

	Vecf_3 cur_curvature;
	cur_curvature.setZero();
	for (int i = 0; i < coeffs.size(); i++) {
		cur_curvature += coeffs[i] * verts.row(m_bodies[i]);
	}
	Real C = cur_curvature.squaredNorm() - rest_mean_curvature_norm;

	std::vector<Vecf_3> grad_c(this->m_bodies.size());
	Real sum_normgradC = 0;
	for (int i = 0; i < this->m_bodies.size(); i++) {
		grad_c[i] = 2 * cur_curvature * coeffs[i];
		sum_normgradC += inv_mass(m_bodies[i]) * grad_c[i].squaredNorm();
	}

	Real grad_c_beta = 0.0_r;
	if (beta.rows() > 0) {
		for (int i = 0; i < this->m_bodies.size(); i++) {
			grad_c_beta += grad_c[i].dot(beta.row(m_bodies[i]));
		}
	}

	if (iter == 0) m_lambda = 0.0_r;

	if (fabs(sum_normgradC) > XPBD_utils::eps) {
		Real alpha = 1.0_r / (stiffness * dt * dt);
		std::vector<Vecf_3> corr_x(this->m_bodies.size());

		//Real s = (C) / (sum_normgradC + alpha);
		Real s = (C + alpha * m_lambda - grad_c_beta) / (sum_normgradC + alpha);
		m_lambda -= s;

		for (int i = 0; i < this->m_bodies.size(); i++) {
			Vecf_3 corr_xi = -(s * inv_mass(m_bodies[i])) * grad_c[i];
			verts.row(m_bodies[i]) += corr_xi;
		}

		/*if (beta.rows() > 0) {
			for (int i = 0; i < this->m_bodies.size(); i++) {
				Vecf_3 corr_xi = -(s * inv_mass(m_bodies[i])) * grad_c[i] - beta.row(i).transpose();
				verts.row(m_bodies[i]) += corr_xi;
			}
		}
		else {
			for (int i = 0; i < this->m_bodies.size(); i++) {
				Vecf_3 corr_xi = -(s * inv_mass(m_bodies[i])) * grad_c[i];
				verts.row(m_bodies[i]) += corr_xi;
			}
		}*/
	}
	/*else {
		if (beta.rows() > 0) {
			for (int i = 0; i < this->m_bodies.size(); i++) {
				Vecf_3 corr_xi = - beta.row(i).transpose();
				verts.row(m_bodies[i]) += corr_xi;
			}
		}
	}*/
	return true;
}