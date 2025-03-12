//
// Created by Siyan on 2024/10/20.
//

#ifndef TETRAHEDRAL_CONSTRAINT_CU_H
#define TETRAHEDRAL_CONSTRAINT_CU_H

#include "mutils/common_types.h"
#include "constraint/tetrahedral_constraint.h"
#include <thrust/host_vector.h>
#include "thrust/device_vector.h"



class TetrahedralConstraintDevice {
public:
    thrust::device_vector<ADU::Real> min_limit; // Minimum strain limits
    thrust::device_vector<ADU::Real> max_limit; // Maximum strain limits
    thrust::device_vector<int> use_limit;  // Whether to use strain limits
    thrust::device_vector<ADU::Real> k;     // Stiffness coefficient
    thrust::device_vector<ADU::Real> w;     // Weight coefficient based on stiffness and area

    int num_constraints;
    int dim;

    explicit TetrahedralConstraintDevice(const std::vector<std::shared_ptr<TetrahedralConstraint>>& constraints);

    TetrahedralConstraintDevice(const TetrahedralConstraintDevice& other) = delete;

    TetrahedralConstraintDevice& operator=(const TetrahedralConstraintDevice& other) = delete;

    void run_proxy(ADU::Real* zi);

};




#endif //TETRAHEDRAL_CONSTRAINT_CU_H
