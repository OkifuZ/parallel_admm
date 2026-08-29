#pragma once

#include "solver/core/admm_backend.h"
#include "solver/backend_gpu/admm_impl_gpu.h"

namespace ADU {

/// GPU elastic prox: D*x + Ue assembled and proxed by the per-type device
/// kernels (tet / triangle / bending / pin run_proxy).
class LocalProjectorGPU : public ILocalProjector {
public:
    explicit LocalProjectorGPU(ADMMImplGPU* impl) : impl_(impl) {}

    void project_elastic() override { impl_->do_project_elastic(); }

private:
    ADMMImplGPU* impl_;
};

} // namespace ADU
