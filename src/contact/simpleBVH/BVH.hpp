#pragma once

#include "contact/simpleBVH/Morton.hpp"
#include "mutils/common_types.h"
#include <Eigen/Core>

#include <vector>
#include <array>
#include <cassert>

#include <tbb/concurrent_vector.h>



namespace SimpleBVH {

using VectorMax3d =
    Eigen::Matrix<ADU::Real, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;

class BVH {
public:
    void init(const std::vector<std::array<ADU::Vecf_3, 2>>& cornerlist);

    void
    init(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F, const ADU::Real tol);

    void clear()
    {
        boxlist.clear();
        new2old.clear();
        n_corners = -1;
    }

    void intersect_3D_box(
        const ADU::Vecf_3& bbd0,
        const ADU::Vecf_3& bbd1,
        std::vector<unsigned int>& list) const
    {
        std::vector<unsigned int> tmp;
        assert(n_corners >= 0);
        box_search_recursive(bbd0, bbd1, tmp, 1, 0, n_corners);

        list.resize(tmp.size());
        for (int i = 0; i < tmp.size(); ++i)
            list[i] = new2old[tmp[i]];
    }

    void intersect_3D_box_tbb(
        const ADU::Vecf_3& bbd0,
        const ADU::Vecf_3& bbd1,
        tbb::concurrent_vector<unsigned int>& list) const
    {
        std::vector<unsigned int> tmp;
        assert(n_corners >= 0);
        box_search_recursive(bbd0, bbd1, tmp, 1, 0, n_corners);

        list.resize(tmp.size());
        for (int i = 0; i < tmp.size(); ++i)
            list[i] = new2old[tmp[i]];
    }

    void intersect_2D_box(
        const ADU::Vecf_2& bbd0,
        const ADU::Vecf_2& bbd1,
        std::vector<unsigned int>& list) const
    {
        ADU::Vecf_3 bbd0_3D = ADU::Vecf_3::Zero();
        bbd0_3D.head<2>() = bbd0;

        ADU::Vecf_3 bbd1_3D = ADU::Vecf_3::Zero();
        bbd1_3D.head<2>() = bbd1;

        intersect_3D_box(bbd0_3D, bbd1_3D, list);
    }

    void intersect_box(
        const VectorMax3d& bbd0,
        const VectorMax3d& bbd1,
        std::vector<unsigned int>& list) const
    {
        ADU::Vecf_3 bbd0_3D = ADU::Vecf_3::Zero();
        bbd0_3D.head(bbd0.size()) = bbd0.head(bbd0.size());

        ADU::Vecf_3 bbd1_3D = ADU::Vecf_3::Zero();
        bbd1_3D.head(bbd1.size()) = bbd1.head(bbd1.size());

        intersect_3D_box(bbd0_3D, bbd1_3D, list);
    }

    void intersect_box_tbb(
        const VectorMax3d& bbd0,
        const VectorMax3d& bbd1,
        tbb::concurrent_vector<unsigned int>& list) const
    {
        ADU::Vecf_3 bbd0_3D = ADU::Vecf_3::Zero();
        bbd0_3D.head(bbd0.size()) = bbd0.head(bbd0.size());

        ADU::Vecf_3 bbd1_3D = ADU::Vecf_3::Zero();
        bbd1_3D.head(bbd1.size()) = bbd1.head(bbd1.size());

        intersect_3D_box_tbb(bbd0_3D, bbd1_3D, list);
    }

private:
    void init_boxes_recursive(
        const std::vector<std::array<ADU::Vecf_3, 2>>& cornerlist,
        int node_index,
        int b,
        int e);

    void box_search_recursive(
        const ADU::Vecf_3& bbd0,
        const ADU::Vecf_3& bbd1,
        std::vector<unsigned int>& list,
        int n,
        int b,
        int e) const;

    void box_search_recursive_tbb(
        const ADU::Vecf_3& bbd0,
        const ADU::Vecf_3& bbd1,
        std::vector<unsigned int>& list,
        int n,
        int b,
        int e) const {};

    bool box_intersects_box(
        const ADU::Vecf_3& bbd0,
        const ADU::Vecf_3& bbd1,
        int index) const;

    static int max_node_index(int node_index, int b, int e);

    Eigen::Matrix<ADU::Real, Eigen::Dynamic, 3, Eigen::RowMajor> box_centers;

    struct sortstruct {
        int order;
        Resorting::MortonCode64 morton;
    };
    std::vector<sortstruct> list;

    std::vector<std::array<ADU::Vecf_3, 2>> boxlist;
    std::vector<int> new2old;
    std::vector<std::array<ADU::Vecf_3, 2>> sorted_cornerlist;
    size_t n_corners = -1;
};
} // namespace SimpleBVH
