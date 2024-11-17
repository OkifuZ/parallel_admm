//
// Created by Siyan on 2024/10/20.
//

#ifndef PIN_CONSTRAINT_CU_H
#define PIN_CONSTRAINT_CU_H

#include "mutils/common_types.h"
#include <thrust/host_vector.h>
#include "thrust/device_vector.h"
#include "constraint/pin_constraint.h"



class PinConstraintDevice {
public:
    thrust::device_vector<ADU::Real> pin_position;  // Flattened 2x2 matrices (4 elements per constraint)

    thrust::device_vector<ADU::Real> k;     // Stiffness coefficient
    thrust::device_vector<ADU::Real> w;     // Weight coefficient based on stiffness and area
    thrust::device_vector<int> global_m;    // Global index
    thrust::device_vector<int> start_row;   // Start row for block operations
    thrust::device_vector<int> inds;        // Vertex indices (flattened array, size = 3 * num_constraints)

    int num_constraints;
    int dim;

    void init(std::vector<ADU::Vecf_3> const& pin_position, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds);

    explicit PinConstraintDevice(
        std::vector<ADU::Vecf_3> const& pin_position, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds);

    explicit PinConstraintDevice(const std::vector<std::shared_ptr<PinConstraint>>& constraints);

    PinConstraintDevice(const PinConstraintDevice &other) = delete;
    PinConstraintDevice &operator=(const PinConstraintDevice &other) = delete;

    void run_proxy(ADU::Real* zi);

};




#endif //PIN_CONSTRAINT_CU_H
