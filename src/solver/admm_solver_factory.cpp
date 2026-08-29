#include "solver/admm_solver_factory.h"
#include "solver/backend_cpu/admm_backend_cpu.h"
#include "solver/backend_gpu/admm_backend_gpu.h"

namespace ADU {

std::unique_ptr<IADMMBackend> create_admm_backend(bool use_gpu) {
    if (use_gpu)
        return std::make_unique<ADMMBackendGPU>();
    return std::make_unique<ADMMBackendCPU>();
}

} // namespace ADU
