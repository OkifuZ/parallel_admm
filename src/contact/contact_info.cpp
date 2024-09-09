#include "contact/contact_info.h"

ADU::Real ContactParameter::thickness = static_cast<ADU::Real>(5e-2);
ADU::Real ContactParameter::radius = static_cast<ADU::Real>(7e-2);

ADU::Real ContactParameter::precision = std::numeric_limits<ADU::Real>::epsilon() * 10;