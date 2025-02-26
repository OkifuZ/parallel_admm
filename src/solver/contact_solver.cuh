#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact);

