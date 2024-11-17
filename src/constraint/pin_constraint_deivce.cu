//
// Created by Siyan on 2024/10/20.
//

#include "pin_constraint.h"
#include "constraint/pin_constraint_device.h"
#include "thrust/execution_policy.h"
#include "mutils/svd_fast3.cuh"

void PinConstraintDevice::init(std::vector<ADU::Vecf_3> const& pin_position, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds)
{
    this->pin_position.resize(pin_position.size() * 3);
    thrust::copy(
        (pin_position.data()->data()),
        (pin_position.data()->data()) + pin_position.size() * 3,
        this->pin_position.begin());

    this->k = k;
    this->w = w;
    this->global_m = global_m;
    this->start_row = start_row;
    this->inds = inds; // n * 3

    num_constraints = pin_position.size();
}


PinConstraintDevice::PinConstraintDevice(
    std::vector<ADU::Vecf_3> const& pin_position, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
    std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds)
    : dim(1)
{
    init(pin_position, k, w, global_m, start_row, inds);

    num_constraints = pin_position.size();
}


PinConstraintDevice::PinConstraintDevice(const std::vector<std::shared_ptr<PinConstraint>>& constraints)
    : dim(1)
{
    num_constraints = constraints.size();

    std::vector<ADU::Vecf_3> pin_position(num_constraints);
    std::vector<ADU::Real> k(num_constraints);
    std::vector<ADU::Real> w(num_constraints);
    std::vector<int> global_m(num_constraints);
    std::vector<int> start_row(num_constraints);
    std::vector<int> inds(num_constraints * 3);

    for (int i = 0; i < num_constraints; i++) {
        auto& constraint = constraints[i];
        pin_position[i] = constraint->pin_poistion;

        k[i] = constraint->k;
        w[i] = constraint->w;
        global_m[i] = constraint->global_m;
        start_row[i] = constraint->start_row;
        inds[i] = constraint->inds[0];
    }
    init(pin_position, k, w, global_m, start_row, inds);
}


// TODO handle **start row** , in host it is handled out side the prox foo

// Define a struct to perform the prox operation in a parallel manner.
struct PinProxFunctor {
    const ADU::Real* pin_position;

    ADU::Real* zi;  // Flattened zi, each constraint is stored as a flattened 2x3 matrix.
    int num_constraints;
    int dim;

    PinProxFunctor(
        const ADU::Real* pin_position,
        ADU::Real* zi, // m*3 -> 3m*1
        int num_constraints,
        int dim
    ) : pin_position(pin_position), zi(zi), num_constraints(num_constraints), dim(dim) {}

    __device__ void operator()(int idx) const {
        using namespace Eigen;
        using namespace ADU;

        ADU::Real pin_x = pin_position[idx * 3 + 0];
        ADU::Real pin_y = pin_position[idx * 3 + 1];
        ADU::Real pin_z = pin_position[idx * 3 + 2];

        // TODO start row
        ADU::Real* zi_block = zi + idx * dim * 3; // (1*3)^T
        zi_block[0] = pin_x;
        zi_block[1] = pin_y;
        zi_block[2] = pin_z;
    }
};

// Function to run the parallel prox operation on the constraints.
void PinConstraintDevice::run_proxy(ADU::Real* zi) {
    PinProxFunctor functor(
        thrust::raw_pointer_cast(this->pin_position.data()),
        zi,
        this->num_constraints,
        this->dim
    );

    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(this->num_constraints),
        functor
    );
}

