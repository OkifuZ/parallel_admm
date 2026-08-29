#include "solver/core/admm_solver.h"

namespace ADU {

void ADMMSolver::init(const Matf_X3& verts, const Vecf_X& mass, Real dt) {
    if (backend_ && backend_->as_solver())
        backend_->as_solver()->init(verts, mass, dt);
}

void ADMMSolver::step() {
    if (backend_) backend_->step();
}

void ADMMSolver::reset(const Matf_X3& ini_verts, bool need_precompute) {
    if (backend_ && backend_->as_solver())
        backend_->as_solver()->reset(ini_verts, need_precompute);
}

const Matf_X3& ADMMSolver::getVertices() const {
    return backend_->as_solver()->getVertices();
}

Matf_X3& ADMMSolver::getVertices() {
    return backend_->as_solver()->getVertices();
}

const Matf_X3& ADMMSolver::getVelocities() const {
    return backend_->as_solver()->getVelocities();
}

Matf_X3& ADMMSolver::getVelocities() {
    return backend_->as_solver()->getVelocities();
}

} // namespace ADU
