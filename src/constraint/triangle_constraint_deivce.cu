//
// Created by Siyan on 2024/10/20.
//

#include "constraint/triangle_constraint_device.h"
#include "thrust/execution_policy.h"
#include "mutils/svd_fast3.cuh"

__device__
void SVD_impl(const ADU::Matf_33& A_, ADU::Matf_33& U_, ADU::Vecf_3& S_, ADU::Matf_33& V_) {

    Eigen::Matrix3f A = A_.cast<float>();
    Eigen::Matrix3f U = U_.cast<float>();
    Eigen::Matrix3f V = V_.cast<float>();
    Eigen::Vector3f S = S_.cast<float>();
    svd(
        A(0, 0), A(0, 1), A(0, 2),
        A(1, 0), A(1, 1), A(1, 2),
        A(2, 0), A(2, 1), A(2, 2),
        U(0, 0), U(0, 1), U(0, 2),
        U(1, 0), U(1, 1), U(1, 2),
        U(2, 0), U(2, 1), U(2, 2),
        S(0), S( 1), S(2),
        V(0, 0), V(0, 1), V(0, 2),
        V(1, 0), V(1, 1), V(1, 2),
        V(2, 0), V(2, 1), V(2, 2));

    U_ = U.cast<double>();
    V_ = V.cast<double>();
    S_ = S.cast<double>();
}


void TriangleConstraintDevice::init(std::vector<ADU::Matf_22> const& binv, std::vector<ADU::Real> const& area,
        std::vector<ADU::Real> const& min_limit, std::vector<ADU::Real> const& max_limit,
        std::vector<ADU::Real> const& use_limit, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds)
{
    this->Binv.resize(binv.size() * 4);
    thrust::copy(
        (binv.data()->data()),
        (binv.data()->data()) + binv.size() * 4,
        this->Binv.begin());
    this->area = area;
    this->min_limit = min_limit;
    this->max_limit = max_limit;
    this->use_limit = use_limit;
    this->k = k;
    this->w = w;
    this->global_m = global_m;
    this->start_row = start_row;
    this->inds = inds; // n * 3

    num_constraints = binv.size();
}


TriangleConstraintDevice::TriangleConstraintDevice(
        std::vector<ADU::Matf_22> const& binv, std::vector<ADU::Real> const& area,
        std::vector<ADU::Real> const& min_limit, std::vector<ADU::Real> const& max_limit,
        std::vector<ADU::Real> const& use_limit, std::vector<ADU::Real> const& k, std::vector<ADU::Real> const& w,
        std::vector<int> const& global_m, std::vector<int> const& start_row, std::vector<int> const& inds)
    : dim(2)
{
    init(binv, area, min_limit, max_limit, use_limit, k, w, global_m, start_row, inds);

    num_constraints = binv.size();
}


TriangleConstraintDevice::TriangleConstraintDevice(const std::vector<std::shared_ptr<TriangleConstraint>>& constraints)
    : dim(2)
{
    num_constraints = constraints.size();

    std::vector<ADU::Matf_22> binv(num_constraints);
    std::vector<ADU::Real> area(num_constraints);
    std::vector<ADU::Real> min_limit(num_constraints);
    std::vector<ADU::Real> max_limit(num_constraints);
    std::vector<ADU::Real> use_limit(num_constraints);
    std::vector<ADU::Real> k(num_constraints);
    std::vector<ADU::Real> w(num_constraints);
    std::vector<int> global_m(num_constraints);
    std::vector<int> start_row(num_constraints);
    std::vector<int> inds(num_constraints * 3);

    for (int i = 0; i < num_constraints; i++) {
        auto& constraint = constraints[i];
        binv[i] = constraint->Binv;
        area[i] = constraint->area;
        min_limit[i] = constraint->min_limit;
        max_limit[i] = constraint->max_limit;
        k[i] = constraint->k;
        w[i] = constraint->w;
        global_m[i] = constraint->global_m;
        start_row[i] = constraint->start_row;
        inds[i * 3 + 0] = constraint->inds[0];
        inds[i * 3 + 1] = constraint->inds[1];
        inds[i * 3 + 2] = constraint->inds[2];
    }
    init(binv, area, min_limit, max_limit, use_limit, k, w, global_m, start_row, inds);
}




// Define a struct to perform the prox operation in a parallel manner.
struct TriProxFunctor {
    const ADU::Real* Binv;
    const ADU::Real* area;
    const ADU::Real* min_limit;
    const ADU::Real* max_limit;
    const int* use_limit;
    ADU::Real* zi;  // Flattened zi, each constraint is stored as a flattened 2x3 matrix.
    int num_constraints;
    int dim;

    TriProxFunctor(
        const ADU::Real* Binv,
        const ADU::Real* area,
        const ADU::Real* min_limit,
        const ADU::Real* max_limit,
        const int* use_limit,
        ADU::Real* zi,
        int num_constraints,
        int dim
    ) : Binv(Binv), area(area), min_limit(min_limit), max_limit(max_limit),
        use_limit(use_limit), zi(zi), num_constraints(num_constraints), dim(dim) {}

    __device__ void operator()(int idx) const {
        using namespace Eigen;
        using namespace ADU;

        // zi is stored as a flattened array, so we need to extract the relevant part.
        // Each zi is a 2x3 block, so we calculate the offset.
        ADU::Real* zi_block = zi + idx * dim * 3; // (2*3)^T

        // Map the flattened block back into Eigen matrices for easier manipulation.
        Matrix<ADU::Real, 3, 2> F;
        F.col(0) = ADU::Vecf_3(zi_block[0], zi_block[1], zi_block[2]);
        F.col(1) = ADU::Vecf_3(zi_block[3], zi_block[4], zi_block[5]);

        // Perform SVD.

        ADU::Matf_33 A, U, V, S_;
        A.setZero();
        A.block(0, 0, 3, 2) = F;
        ADU::Vecf_3 S;
        SVD_impl(A, U, S, V);
        S_.setZero();


        if (abs(S(0)) < 1e-8) {
            S_(0, 0) = 0;
            S_(1, 1) = 1;
            S_(2, 2) = 1;
        }
        else if (abs(S(1)) < 1e-8) {
            S_(0, 0) = 1;
            S_(1, 1) = 0;
            S_(2, 2) = 1;
        }
        else{
            S_(0, 0) = 1;
            S_(1, 1) = 1;
            S_(2, 2) = 0;
        }

        ADU::Matf_33 P_ = U * S_ * V.transpose();

        ADU::Matf_32 P;
        if (abs(S(0)) < 1e-8) {
            P.col(0) = P_.col(1);
            P.col(1) = P_.col(2);
        }
        else if (abs(S(1)) < 1e-8) {
            P.col(0) = P_.col(0);
            P.col(1) = P_.col(2);
        }
        else  {
            P.col(0) = P_.col(0);
            P.col(1) = P_.col(1);
        }


        // Update zi with (0.5 * (P + F)).transpose()
        Matrix<ADU::Real, 2, 3> zi_updated = (0.5f * (P + F)).transpose();


        // Store the updated result back into the flattened array.
        zi_block[0] = zi_updated(0, 0);
        zi_block[1] = zi_updated(0, 1);
        zi_block[2] = zi_updated(0, 2);
        zi_block[3] = zi_updated(1, 0);
        zi_block[4] = zi_updated(1, 1);
        zi_block[5] = zi_updated(1, 2);
    }
};

// Function to run the parallel prox operation on the constraints.
// special caution for Zi
void TriangleConstraintDevice::run_proxy(ADU::Real* zi) {
    TriProxFunctor functor(
        thrust::raw_pointer_cast(this->Binv.data()),
        thrust::raw_pointer_cast(this->area.data()),
        thrust::raw_pointer_cast(this->min_limit.data()),
        thrust::raw_pointer_cast(this->max_limit.data()),
        thrust::raw_pointer_cast(this->use_limit.data()),
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

