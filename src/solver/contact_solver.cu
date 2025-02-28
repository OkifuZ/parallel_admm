#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include "mutils/common_types.h"

#include "solver/contact_solver.cuh"


using namespace  ADU;

__global__ void updateContactCounts(const int4* contacts, int* vi_ct_nums, int nContact) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nContact) {
        const auto& c = contacts[idx];

        // Use atomicAdd to update the counts for each vertex index
        atomicAdd(&vi_ct_nums[c.x], 1);
        atomicAdd(&vi_ct_nums[c.y], 1);
        atomicAdd(&vi_ct_nums[c.z], 1);
        atomicAdd(&vi_ct_nums[c.w], 1);
    }
}

void computeContactCountsWithAtomic(const int4* d_contacts, thrust::device_vector<int>& d_vi_ct_nums, int nContact) {

    // Launch kernel with a sufficient number of threads per block
    int blockSize = 256; // This can be adjusted
    int numBlocks = (nContact + blockSize - 1) / blockSize;

    updateContactCounts << <numBlocks, blockSize >> > (d_contacts, thrust::raw_pointer_cast(d_vi_ct_nums.data()), nContact);
    cudaDeviceSynchronize(); // Ensure the kernel finishes
}

void compute_Scc_impl_cu(const thrust::device_vector<int>& d_vi_ct_nums,  thrust::device_vector<int>& d_start_idx, thrust::device_vector<int>& d_involved_cid)
{
    d_start_idx.resize(d_vi_ct_nums.size() + 1);
    thrust::inclusive_scan(d_vi_ct_nums.begin() + 1, d_vi_ct_nums.end(), d_start_idx.begin());
    int total_size = d_start_idx[d_start_idx.size() - 1];
    d_involved_cid.resize(total_size, 0);
    printf("device: %d\n", total_size);

}