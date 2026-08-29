#pragma once

#include "solver/core/admm_backend.h"
#include <memory>

namespace ADU {

/// Factory: create ADMM backend by use_gpu flag.
/// Application layer only needs this + IADMMBackend; no impl/backend headers.
std::unique_ptr<IADMMBackend> create_admm_backend(bool use_gpu);

} // namespace ADU
