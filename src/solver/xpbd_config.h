#pragma once

#include "mutils/common_types.h"

namespace ADU {

/// XPBD tuning parameters, derived from the [xpbd] TOML section by the app
/// layer and applied to XPBDSolver. Plain data: no solver/impl types.
struct XPBDConfig {
    unsigned int substeps{ 1 };
    unsigned int max_iter{ 400 };
    ADU::Real contact_stiffness{ 10 };
    bool use_GS_contact{ true };
    ADU::Real g{ static_cast<ADU::Real>(0.98) };

    // --- frictional contact ---
    bool enable_frictional_contact{ true };
    int dcd_interval{ 5 };
    bool use_unique_contact{ false };
    ADU::Real mu{ static_cast<ADU::Real>(0.5) };
};

} // namespace ADU
