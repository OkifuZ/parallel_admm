//#pragma once
//#include "mutils/common_types.h"
//
////#define EIGEN_USE_MKL_ALL
////#include "Eigen/Eigen"
////#include <Eigen/PardisoSupport>
//
//#include "solver/solver.h"
//#include "constraint/constraint.h"
//#include "constraint/tetrahedral_constraint.h"
//
//#include "contact/narrow_phase.h"
//
//#include <tbb/concurrent_vector.h>
//#include <mutex>
//#include <memory>
//#include <random>
//#include "mutils/common_types.h"
//using namespace ADU;
//
//
//class PBDSolver : public Solver {
//public:
//	Vecf_X m_vec;
//	Real volumeStiffness = 100.0;
//	Real g{ static_cast <ADU::Real>(0.98) };
//	virtual void step();
//	void solveVolume(float sdt, const std::shared_ptr<Constraint> ct, Matf_X3& x_curr);
//	void solvePin(float sdt, const std::shared_ptr<Constraint> ct, Matf_X3& x_curr);
//};
//
//void PBDSolver::step() {
//	size_t pbd_max_iter = 20;
//
//	float sdt = 1.0f * m_dt / pbd_max_iter;
//	for (size_t pbd_it = 0; pbd_it < pbd_max_iter; pbd_it++) {
//		// explicit integrate as initial value
//		for (int i = 0; i < m_nVert; i++) {
//			if (m_pin_inds_set.count(i)) continue;
//			if (m_vStatic(i)) continue;
//			m_velocities.row(i).y() -= g * sdt;
//		}
//		Matf_X3 x_0(m_vertices);
//		Matf_X3 v_0(m_velocities);
//		Matf_X3 x_curr = x_0 + sdt * v_0;
//		for (size_t ci = 0; ci < m_constraints.size() - m_nVert; ci++) {
//			const auto& ct = m_constraints[ci];
//			if (ct->type == 2) {
//				//tet
//				solveVolume(sdt, ct, x_curr);
//			}
//			if (ct->type == 4) {
//				solvePin(sdt, ct, x_curr);
//			}
//		}
//		m_velocities = (x_curr - x_0) / sdt;
//	}
//}
//
//void PBDSolver::solvePin(float sdt, const std::shared_ptr<Constraint> ct, Matf_X3& x_curr) {
//
//}
//
//void PBDSolver::solveVolume(float sdt, const std::shared_ptr<Constraint> ct, Matf_X3& x_curr) {
//	auto tetp = std::static_pointer_cast<TetrahedralConstraint>(ct);
//	ADU::Real restV = tetp->volume;
//	int v0 = ct->inds[0];
//	int v1 = ct->inds[1];
//	int v2 = ct->inds[2];
//	int v3 = ct->inds[3];
//	Vecf_3 p0 = x_curr.row(v0);
//	Vecf_3 p1 = x_curr.row(v1);
//	Vecf_3 p2 = x_curr.row(v2);
//	Vecf_3 p3 = x_curr.row(v3);
//	Real volume = (p1 - p0).cross(p2 - p0).dot(p3 - p0);
//	Real C = volume - restV;
//
//	Vecf_3 grad0 = (p3 - p1).cross(p2 - p1);
//	Vecf_3 grad1 = (p2 - p0).cross(p3 - p0);
//	Vecf_3 grad2 = (p3 - p0).cross(p1 - p0);
//	Vecf_3 grad3 = (p1 - p0).cross(p2 - p0);
//
//	Real w0 = 1.0f / m_vec[v0];
//	Real w1 = 1.0f / m_vec[v1];
//	Real w2 = 1.0f / m_vec[v2];
//	Real w3 = 1.0f / m_vec[v3];
//
//	Real alpha = 1.0f / volumeStiffness;
//
//	Real lambda = -C /
//		(w0 * grad0.dot(grad0) + w1 * grad1.dot(grad1) +
//			w2 * grad2.dot(grad2) + w3 * grad3.dot(grad3) + alpha / (sdt * sdt));
//
//	x_curr.row(v0) += lambda * w0 * grad0;
//	x_curr.row(v1) += lambda * w1 * grad1;
//	x_curr.row(v2) += lambda * w2 * grad2;
//	x_curr.row(v3) += lambda * w3 * grad3;
//}
//
//#pragma once
