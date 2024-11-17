#pragma once

#include "compact_sparse_matrix.h"


void CompactSparseMat::buildSystemMatrix(const ADU::SpMatf &mat) {
    for (int k = 0; k < mat.outerSize(); ++k) {
        int line_vals_num = 0;

        for (ADU::SpMatf::InnerIterator it(mat, k); it; ++it) {
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

