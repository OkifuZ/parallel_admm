#pragma once

#include "solver/core/admm_backend.h"
#include "solver/backend_cpu/admm_impl_cpu.h"

#include <tbb/parallel_for.h>

namespace ADU {

/// CPU elastic prox: z = D*x + Ue, then per-constraint prox.
/// Constraints are contiguous per type; each type range is one parallel
/// range and prox works directly on views of z (no per-constraint copies).
class LocalProjectorCPU : public ILocalProjector {
public:
    explicit LocalProjectorCPU(ADMMImplCPU* impl) : impl_(impl) {}

    void project_elastic() override {
        auto* s = impl_;
        const int nDynVert = s->static_vert_begin;
        s->DX = s->m_D * s->x_curr.block(0, 0, nDynVert, 3);
        s->z = s->DX + s->m_Ue;
        for (const auto& range : s->constraint_type_ranges) {
            tbb::parallel_for(tbb::blocked_range<size_t>(range.first, range.second),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t ci = r.begin(); ci < r.end(); ci++) {
                        const auto& ct = s->m_constraints[ci];
                        const int cdim = ct->dim;
                        ADU::Matf_XX zi = s->z.block(ct->start_row, 0, cdim, 3);
                        ct->prox(zi);
                        s->z.block(ct->start_row, 0, cdim, 3) = zi;
                    }
                });
        }
    }

private:
    ADMMImplCPU* impl_;
};

} // namespace ADU
