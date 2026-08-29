#pragma once

#include "solver/core/admm_backend.h"
#include "solver/core/admm_solver.h"
#include "solver/xpbd_solver.h"
#include <memory>

namespace ADU {

/// Factory: create ADMM backend by use_gpu flag.
/// Application layer only needs this + IADMMBackend; no impl/backend headers.
std::unique_ptr<IADMMBackend> create_admm_backend(bool use_gpu);

/// Unified solver creation entry. Extensible: future solver types
/// (projective dynamics, ...) get a slot here without touching the
/// application layer.
enum class SolverType { ADMM_CPU, ADMM_GPU, XPBD };

std::unique_ptr<Solver> create_solver(SolverType type);

} // namespace ADU
