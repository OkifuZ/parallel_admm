//
// Created by Siyan on 2024/10/19.
//

#ifndef COMMON_TYPE_HOSTONLY_H
#define COMMON_TYPE_HOSTONLY_H

#include "mutils/common_types.h"
#include "Eigen/Sparse"

namespace ADU {

    using SpMatf = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
    using Tripf = Eigen::Triplet<Real>;
}

#endif //COMMON_TYPE_HOSTONLY_H
