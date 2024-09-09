#include "mutils/common_types.h"
#include <vector>
#include <iostream>


struct CompactSparseMat{
    std::vector<ADU::Real> diag_data;
    std::vector<ADU::Real> offdiag_data;
    std::vector<int> offdiag_indices;
    std::vector<int> offdiag_perline_start; // [), size = n+1

    CompactSparseMat(const ADU::SpMatf& mat) {
        
        // each line has at least 1 value
        offdiag_perline_start.push_back(0);
        using namespace ADU;
        int n = mat.cols();
        int m = mat.rows();
        for (int k = 0; k < mat.outerSize(); ++k) {
            int line_vals_num = 0;
            for (SpMatf::InnerIterator it(mat, k); it; ++it) {
                int col = it.col();
                int row = it.row();
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