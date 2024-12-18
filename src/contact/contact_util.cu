#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/zip_iterator.h>
#include <Eigen/Core>


#include "contact/contact_util.cuh"


// Custom comparator for int4 to enable sorting
struct Int4Comparator {
    __host__ __device__
    bool operator()(const int4& a, const int4& b) const {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y) ||
               (a.x == b.x && a.y == b.y && a.z < b.z) ||
               (a.x == b.x && a.y == b.y && a.z == b.z && a.w < b.w);
    }
};

// Custom equality operator for int4
struct Int4Equality {
    __host__ __device__
    bool operator()(const int4& a, const int4& b) const {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }
};

// Functor to filter based on barycentric[2] value
template <typename T>
struct BarycentricFilter {
    __host__ __device__
    bool operator()(const Result<T>& res) const {
        return res.barycentric[2] == -5;
    }
};

size_t removeDuplicates(int4* d_collisionPairs, Result<ADU::Real>* d_contact_info, size_t size) {
    // Wrap raw pointers with thrust::device_vector for convenience
    /*thrust::device_vector<int4> collisionPairs(d_collisionPairs, d_collisionPairs + size);
    thrust::device_vector<Result<ADU::Real>> contactInfo(d_contact_info, d_contact_info + size);*/

    thrust::device_vector<int4> collisionPairs(size);
    thrust::device_vector<Result<ADU::Real>> contactInfo(size);

    cudaMemcpy(thrust::raw_pointer_cast(collisionPairs.data()), d_collisionPairs, size * sizeof(int4), cudaMemcpyDeviceToDevice);
    cudaMemcpy(thrust::raw_pointer_cast(contactInfo.data()), d_contact_info, size * sizeof(Result<ADU::Real>), cudaMemcpyDeviceToDevice);


    // Sort collision pairs and corresponding contact info
    thrust::sort_by_key(collisionPairs.begin(), collisionPairs.end(), contactInfo.begin(), Int4Comparator());

    // Remove duplicates using unique
    auto new_end = thrust::unique_by_key(collisionPairs.begin(), collisionPairs.end(),
                                         contactInfo.begin(), Int4Equality());
    // Resize device vectors to remove excess elements
    size_t new_size = thrust::distance(collisionPairs.begin(), new_end.first);
    collisionPairs.resize(new_size);
    contactInfo.resize(new_size);
    // Copy back to raw pointers (if needed)
    /*thrust::copy(collisionPairs.begin(), collisionPairs.end(), d_collisionPairs);
    thrust::copy(contactInfo.begin(), contactInfo.end(), d_contact_info);*/

    cudaMemcpy(d_collisionPairs, raw_pointer_cast(collisionPairs.data()), size * sizeof(int4), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_contact_info, thrust::raw_pointer_cast(contactInfo.data()), size * sizeof(Result<ADU::Real>), cudaMemcpyDeviceToDevice);


    return new_size;
}