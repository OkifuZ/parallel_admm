#pragma once
#include "mutils/common_types.h"
#include <memory>
#include <vector>

class Obstacle {
public:
	int type = 0; // 0: sphere, 1: plane
	ADU::Vecf_3 center;

	Obstacle(int type, const ADU::Vecf_3& center) :
	type(type), center(center) {}

	virtual bool point_in_obstacle(const ADU::Vecf_3& pt) const { return false; }
};

class SphereObstacle : public Obstacle {
public:
	ADU::Real radius{};

	SphereObstacle(const ADU::Vecf_3& center, ADU::Real radius):
		Obstacle(0, center), radius(radius) {}

	virtual bool point_in_obstacle(const ADU::Vecf_3& pt) const { 
		//printf("%f %f\n", (pt - center).squaredNorm(), radius * radius);
		return (pt - center).squaredNorm() < (radius * radius); 
	}

};

class PlaneObstacle : public Obstacle {
public:
	ADU::Real half_edge{};
	ADU::Vecf_3 normal{};

	PlaneObstacle(const ADU::Vecf_3& center, ADU::Real half_edge, const ADU::Vecf_3& normal_) :
		Obstacle(1, center), half_edge(half_edge) 
	{
		using namespace ADU;
		Real nnorm = normal_.norm();
		if (nnorm < 1e-6_r) throw std::runtime_error("Error PlaneObstacle: too small normal");
		else {
			this->normal = normal_ / nnorm;
		}
	}

	virtual bool point_in_obstacle(const ADU::Vecf_3& pt) const {
		return (pt - center).dot(normal) < 0;
	}
};


class ObstacleContactInfo {
public:
	std::shared_ptr<Obstacle> obstacle{};
	int node_ind{};

	ObstacleContactInfo(const std::shared_ptr<Obstacle>& obstacle, int node_ind_) :
		obstacle(obstacle), node_ind(node_ind_)
	{
		if (!obstacle) throw std::runtime_error("Error ContactInfo: empty obstacle");
	}
};

class CollisionDetection {
public:
	using ContactInfoList = std::vector<std::shared_ptr<ObstacleContactInfo>>;
	using ObstacleList = std::vector< std::shared_ptr<Obstacle>>;

	static void collision_detection_nodal_obstacle(
		const ADU::Matf_X3& x, const ObstacleList& obstacles, ContactInfoList& ctinfo_list) 
	{
		using namespace ADU;

		ctinfo_list.clear();

		for (int vi = 0; vi < x.rows(); vi++) {
			const Vecf_3& vx = x.row(vi);
			int cnt = 0;
			for (int oi = 0; oi < obstacles.size(); oi++) {
				auto& obs = obstacles[oi];
				if (cnt >= 1) break;
				if (obs->point_in_obstacle(vx)) {
					cnt++;
					ctinfo_list.emplace_back(std::make_shared<ObstacleContactInfo>(obs, vi));
				}
			}
 		}
	}
};