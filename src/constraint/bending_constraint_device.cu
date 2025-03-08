#include "constraint/bending_constraint_device.cuh"



BendingConstraintDevice::BendingConstraintDevice(const std::vector<std::shared_ptr<BendingConstraint>>& constraints) : dim(1) 
{
	num_constraints = constraints.size();

	std::vector<float3> rest_mean_curvature(num_constraints);
	std::vector<float> res_mean_curvature_norm(num_constraints);
	std::vector<float> k(num_constraints);
	std::vector<float> w(num_constraints);

	for (int i = 0; i < num_constraints; i++) {
		auto& constraint = constraints[i];

		rest_mean_curvature[i] = float3{
			constraint->rest_mean_curvature.x(),
			constraint->rest_mean_curvature.y(),
			constraint->rest_mean_curvature.z() };
		res_mean_curvature_norm[i] = constraint->rest_mean_curvature_norm;
		k[i] = constraint->k;
		w[i] = constraint->w;
	}
	//this->rest_mean_curvature = rest_mean_curvature;
	this->res_mean_curvature_norm = res_mean_curvature_norm;
	this->k = k;
	this->w = w;
}

struct BendProxFunctor 
{
	//const float3* rest_mean_curvature;
	const float* res_mean_curvature_norm;
	const float* k;
	const float* w;
	int dim;
	int num_constraints;
	ADU::Real* zi;

	BendProxFunctor(
		//const float3* rest_mean_curvature,
		const float* res_mean_curvature_norm,
		const float* k,
		const float* w,
		int dim,
		int num_constraints,
		ADU::Real* zi) : 
		//rest_mean_curvature(rest_mean_curvature), 
		res_mean_curvature_norm(res_mean_curvature_norm), 
		k(k), w(w), dim(dim), num_constraints(num_constraints), zi(zi) {}


	__device__ void operator()(int idx) {
		using namespace ADU;

		ADU::Real* zi_block = zi + idx * dim * 3; // (2*3)^T
		Vecf_3 z{ zi_block[0], zi_block[1], zi_block[2] };

		Vecf_3 p = z;
		Real pnorm = p.norm();
		Real rest_norm = res_mean_curvature_norm[idx];

		
		if (pnorm > 1e-7f) {
			p *= rest_norm / pnorm;
		}
		else {
			p = Vecf_3::Zero();
		}

		z = (z + p) / 2.0f;

		zi_block[0] = z.x();
		zi_block[1] = z.y();
		zi_block[2] = z.z();
	}
};


void BendingConstraintDevice::run_proxy(ADU::Real* zi) 
{
	BendProxFunctor functor(
		thrust::raw_pointer_cast(this->res_mean_curvature_norm.data()), 
		thrust::raw_pointer_cast(this->k.data()),
		thrust::raw_pointer_cast(this->w.data()),
		this->dim, this->num_constraints, zi
		);

	thrust::for_each(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(this->num_constraints),
		functor
	);
}