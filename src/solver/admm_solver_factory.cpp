#include "solver/admm_solver_factory.h"
#include "solver/backend_cpu/admm_backend_cpu.h"
#include "solver/backend_gpu/admm_backend_gpu.h"
#include "solver/xpbd_solver.h"

namespace ADU {

std::unique_ptr<IADMMBackend> create_admm_backend(bool use_gpu) {
    if (use_gpu)
        return std::make_unique<ADMMBackendGPU>();
    return std::make_unique<ADMMBackendCPU>();
}

std::unique_ptr<Solver> create_solver(SolverType type) {
    switch (type) {
    case SolverType::ADMM_GPU:
        return std::make_unique<ADMMSolver>(create_admm_backend(true));
    case SolverType::XPBD:
        return std::make_unique<XPBDSolver>();
    case SolverType::ADMM_CPU:
    default:
        return std::make_unique<ADMMSolver>(create_admm_backend(false));
    }
}

} // namespace ADU
