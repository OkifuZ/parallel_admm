#ifndef CONTACT_UTIL
#define CONTACT_UTIL
#include "mutils/dist_g.cuh"
#include "mutils/common_types.h"


size_t removeDuplicates(int4* d_collisionPairs, Result<ADU::Real>* d_contact_info, size_t size);



#endif