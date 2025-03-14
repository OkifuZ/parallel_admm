#include "constraint/tetrahedral_constraint_device.cuh"
#include "mutils/svd_fast3.cuh"


__device__
void SVD_impl_tet(const ADU::Matf_33& A_, ADU::Matf_33& U_, ADU::Vecf_3& S_, ADU::Matf_33& V_) {

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
        S(0), S(1), S(2),
        V(0, 0), V(0, 1), V(0, 2),
        V(1, 0), V(1, 1), V(1, 2),
        V(2, 0), V(2, 1), V(2, 2));

    /*U_ = U.cast<double>();
    V_ = V.cast<double>();
    S_ = S.cast<double>();*/
    U_ = U;
    V_ = V;
    S_ = S;
}

TetrahedralConstraintDevice::TetrahedralConstraintDevice(const std::vector<std::shared_ptr<TetrahedralConstraint>>& constraints) : dim(3)
{
    num_constraints = constraints.size();

    std::vector<ADU::Real> min_limit(num_constraints);
    std::vector<ADU::Real> max_limit(num_constraints);
    std::vector<ADU::Real> use_limit(num_constraints);
    std::vector<ADU::Real> k(num_constraints);
    std::vector<ADU::Real> w(num_constraints);
    for (int i = 0; i < num_constraints; i++) {
        auto& constraint = constraints[i];
        min_limit[i] = constraint->min_limit;
        max_limit[i] = constraint->max_limit;
        k[i] = constraint->k;
        w[i] = constraint->w;
    }
    this->min_limit = min_limit;
    this->max_limit = max_limit;
    this->use_limit = use_limit;
    this->k = k;
    this->w = w;
}


struct TetProxFunctor {
    const ADU::Real* min_limit;
    const ADU::Real* max_limit;
    const int* use_limit;
    ADU::Real* zi;  // Flattened zi, each constraint is stored as a flattened 2x3 matrix.
    int num_constraints;
    int dim;

    TetProxFunctor(
        const ADU::Real* min_limit,
        const ADU::Real* max_limit,
        const int* use_limit,
        ADU::Real* zi,
        int num_constraints,
        int dim
    ) : min_limit(min_limit), max_limit(max_limit), use_limit(use_limit), zi(zi), num_constraints(num_constraints), dim(dim) {}

    __device__ void operator() (int idx) const {
        using namespace Eigen;
        using namespace ADU;

        ADU::Real* zi_block = zi + idx * dim * 3; // (2*3)^T
        Matf_33 F;
        F.col(0) = ADU::Vecf_3(zi_block[0], zi_block[1], zi_block[2]);
        F.col(1) = ADU::Vecf_3(zi_block[3], zi_block[4], zi_block[5]);
        F.col(2) = ADU::Vecf_3(zi_block[6], zi_block[7], zi_block[8]);

        ADU::Matf_33 U, V;
        Vecf_3 S_;
        SVD_impl_tet(F, U, S_, V);
        Matf_33 S = Matf_33::Identity();
        //if (F.determinant() < 0) S(2, 2) = -1.0f;
        Matf_33 P = U * S * V.transpose();

        F = 0.5f * (P + F);
        zi_block[0] = F.col(0)(0);
        zi_block[1] = F.col(0)(1);
        zi_block[2] = F.col(0)(2);

        zi_block[3] = F.col(1)(0);
        zi_block[4] = F.col(1)(1);
        zi_block[5] = F.col(1)(2);

        zi_block[6] = F.col(2)(0);
        zi_block[7] = F.col(2)(1);
        zi_block[8] = F.col(2)(2);
    }

};

void TetrahedralConstraintDevice::run_proxy(ADU::Real* zi) 
{
    TetProxFunctor functor(
        thrust::raw_pointer_cast(this->min_limit.data()),
        thrust::raw_pointer_cast(this->max_limit.data()),
        thrust::raw_pointer_cast(this->use_limit.data()),
        zi,
        this->num_constraints, this->dim
    );

    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(this->num_constraints),
        functor
    );

}




