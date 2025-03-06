
//
// Created by Siyan on 2024/10/20.
//

#ifndef BENDING_CONSTRAINT_CU_H
#define BENDING_CONSTRAINT_CU_H

#include "mutils/common_types.h"
#include "constraint/bending_constraint.h"
#include <thrust/host_vector.h>
#include "thrust/device_vector.h"



class BendingConstraintDevice {
public:
    thrust::device_vector<ADU::Real> Binv;  // Flattened 2x2 matrices (4 elements per constraint)
    thrust::device_vector<ADU::Real> area;  // Stores triangle areas
    thrust::device_vector<ADU::Real> min_limit; // Minimum strain limits
    thrust::device_vector<ADU::Real> max_limit; // Maximum strain limits
    thrust::device_vector<int> use_limit;  // Whether to use strain limits
    thrust::device_vector<ADU::Real> k;     // Stiffness coefficient
    thrust::device_vector<ADU::Real> w;     // Weight coefficient based on stiffness and area
    thrust::device_vector<int> global_m;    // Global index
    thrust::device_vector<int> start_row;   // Start row for block operations
    thrust::device_vector<int> inds;        // Vertex indices (flattened array, size = 3 * num_constraints)

    int num_constraints;
    int dim; // 1

    void init(std::vector<ADU::Matf_22> const& binv, std::vector<ADU::Real> const& area,
        std::vector<ADU::Real> const& min_limit, std::vector<ADU::Real> const& max_limit,
        std::vector<ADU::Real> const& use_limit, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds);

    explicit BendingConstraintDevice(
        std::vector<ADU::Matf_22> const& binv, std::vector<ADU::Real> const& area,
        std::vector<ADU::Real> const& min_limit, std::vector<ADU::Real> const& max_limit,
        std::vector<ADU::Real> const& use_limit, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds);

    explicit BendingConstraintDevice(const std::vector<std::shared_ptr<BendingConstraint>>& constraints);

    BendingConstraintDevice(const BendingConstraintDevice& other) = delete;

    BendingConstraintDevice& operator=(const BendingConstraintDevice& other) = delete;

    void run_proxy(ADU::Real* zi);

};




#endif //BENDING_CONSTRAINT_CU_H
