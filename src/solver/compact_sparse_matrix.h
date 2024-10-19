#pragma once

#include "mutils/common_types.h"
#include "mutils/exception_handle.h"
#include "mutils/cformat.h"
#include <mutils/common_type_hostonly.h>

#include <vector>
#include <iostream>


struct CompactSparseMat{
    std::vector<ADU::Real> diag_data; // [Values]
    std::vector<ADU::Real> offdiag_data; // [Values]
    std::vector<int> offdiag_indices; // [inner indices]
    std::vector<int> offdiag_perline_start; // [), size = n+1, [outer Starts]
    int rows{};
    int cols{};

    // equal to SparseMatrix::makeCompressed(), implement for possible manipulation
    CompactSparseMat(const ADU::SpMatf& mat) {
        // mat is row major

        // each line has at least 1 value
        offdiag_perline_start.push_back(0);
        using namespace ADU;
        cols = mat.cols();
        rows = mat.rows();

        for (int k = 0; k < mat.outerSize(); ++k) {
            int line_vals_num = 0;

            for (SpMatf::InnerIterator it(mat, k); it; ++it) {
                int col = it.col();
                int row = it.row();
                if (row != k) {

                    ADU::make_exception(ADU::cformat("CompactSparseMat: each line has at least 1 value, empty in row %d", k));
                }
                if (col == row) {
                    diag_data.push_back(it.value());
                }
                else {
                    line_vals_num += 1;
                    offdiag_data.push_back(it.value());
                    offdiag_indices.push_back(col);
                }
            }
            offdiag_perline_start.push_back(offdiag_perline_start.back() + line_vals_num);
        }
    }
};