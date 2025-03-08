
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
    thrust::device_vector<float> res_mean_curvature_norm;
    thrust::device_vector<ADU::Real> k;     // Stiffness coefficient
    thrust::device_vector<ADU::Real> w;     // Weight coefficient based on stiffness and area
    
    int num_constraints;
    int dim; // 1

    explicit BendingConstraintDevice(const std::vector<std::shared_ptr<BendingConstraint>>& constraints);

    BendingConstraintDevice(const BendingConstraintDevice& other) = delete;

    BendingConstraintDevice& operator=(const BendingConstraintDevice& other) = delete;

    void run_proxy(ADU::Real* zi);

};




#endif //BENDING_CONSTRAINT_CU_H
