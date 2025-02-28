#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact);

void compute_Scc_impl_cu(const thrust::device_vector<int>& d_vi_ct_nums, thrust::device_vector<int>& d_start_idx, thrust::device_vector<int>& d_involved_cid);