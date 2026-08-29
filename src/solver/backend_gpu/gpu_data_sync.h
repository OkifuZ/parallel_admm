#pragma once

/// Phase 4: Centralized Host <-> Device sync for GPU backend.
/// Re-exports copy utilities from jacobi_solver; documents sync points.
///
/// Sync convention:
/// - UPLOAD (host -> device): init(), reset(), step() when animator updates x_0/v_0
/// - DOWNLOAD (device -> host): step() end, write m_vertices/m_velocities
///
/// Call sites in admm_parallel_global_solver.cu:
/// - init(): upload x_0, v_0 (and m_vertices, m_velocities if animator)
/// - reset(): upload m_vertices, m_velocities
/// - step() per-iter: upload x_curr, v_0 when animator (DCD_interval)
/// - step() end: download x_0_device -> m_vertices, v_0_device -> m_velocities
#include "solver/jacobi_solver.h"
