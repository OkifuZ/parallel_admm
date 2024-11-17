/*
//
// Created by Siyan on 2024/10/20.
//

#ifndef QR_REDUCED_H
#define QR_REDUCED_H


#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

template <typename MatrixType>
class ColPivHouseholderQR {
public:
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Index = typename MatrixType::Index;

    ColPivHouseholderQR(MatrixType& matrix)
        : m_qr(matrix),
          m_rows(matrix.rows()),
          m_cols(matrix.cols()),
          m_size(std::min(m_rows, m_cols))
    {
        m_hCoeffs.resize(m_size);
        m_temp.resize(m_cols);
        m_colsTranspositions.resize(m_cols);
        m_colNormsUpdated.resize(m_cols);
        m_colNormsDirect.resize(m_cols);

        computeInPlace();
    }

    const MatrixType& matrixQ() const { return m_qr; }
    const std::vector<Index>& transpositions() const { return m_colsTranspositions; }
    const std::vector<Scalar>& householderCoefficients() const { return m_hCoeffs; }

private:
    MatrixType m_qr;
    Index m_rows, m_cols, m_size;
    std::vector<Scalar> m_hCoeffs;
    std::vector<Scalar> m_temp;
    std::vector<Index> m_colsTranspositions;
    std::vector<RealScalar> m_colNormsUpdated;
    std::vector<RealScalar> m_colNormsDirect;

    void computeInPlace() {
        // Initialize column norms
        for (Index k = 0; k < m_cols; ++k) {
            m_colNormsDirect[k] = m_qr.col(k).norm();
            m_colNormsUpdated[k] = m_colNormsDirect[k];
        }

        RealScalar threshold_helper =
            std::pow(m_colNormsUpdated[maxIndex(m_colNormsUpdated)] * Eigen::NumTraits<RealScalar>::epsilon(), 2)
            / RealScalar(m_rows);
        RealScalar norm_downdate_threshold = std::sqrt(Eigen::NumTraits<RealScalar>::epsilon());

        for (Index k = 0; k < m_size; ++k) {
            // Find column with the largest norm
            Index biggest_col_index = maxIndex(m_colNormsUpdated, k, m_cols);

            // Apply the column transposition
            m_colsTranspositions[k] = biggest_col_index;
            if (k != biggest_col_index) {
                m_qr.col(k).swap(m_qr.col(biggest_col_index));
                std::swap(m_colNormsUpdated[k], m_colNormsUpdated[biggest_col_index]);
                std::swap(m_colNormsDirect[k], m_colNormsDirect[biggest_col_index]);
            }

            // Generate Householder vector and store it below the diagonal
            Scalar beta;
            householderReflector(m_qr.col(k).tail(m_rows - k), m_hCoeffs[k], beta);
            m_qr.coeffRef(k, k) = beta;

            // Apply Householder transformation to the remaining matrix
            applyHouseholderTransformation(k);

            // Update column norms
            updateColumnNorms(k, norm_downdate_threshold);
        }
    }

    void householderReflector(Eigen::Block<MatrixType> x, Scalar& tau, Scalar& beta) {
        RealScalar norm = x.norm();
        beta = (x.coeff(0) >= 0) ? -norm : norm;
        x.coeffRef(0) -= beta;
        tau = RealScalar(2) / (x.squaredNorm());
    }

    void applyHouseholderTransformation(Index k) {
        Index rows = m_rows - k;
        Index cols = m_cols - k - 1;

        if (cols > 0) {
            auto tailVec = m_qr.col(k).tail(rows);
            for (Index j = k + 1; j < m_cols; ++j) {
                m_temp[j] = tailVec.dot(m_qr.col(j).tail(rows));
            }

            Scalar tau = m_hCoeffs[k];
            for (Index j = k + 1; j < m_cols; ++j) {
                m_qr.col(j).tail(rows).noalias() -= tau * m_temp[j] * tailVec;
            }
        }
    }

    void updateColumnNorms(Index k, RealScalar threshold) {
        for (Index j = k + 1; j < m_cols; ++j) {
            if (m_colNormsUpdated[j] != RealScalar(0)) {
                RealScalar temp = std::abs(m_qr.coeff(k, j)) / m_colNormsUpdated[j];
                temp = (RealScalar(1) + temp) * (RealScalar(1) - temp);
                temp = (temp < RealScalar(0)) ? RealScalar(0) : temp;

                RealScalar temp2 = temp * std::pow(m_colNormsUpdated[j] / m_colNormsDirect[j], 2);
                if (temp2 <= threshold) {
                    m_colNormsDirect[j] = m_qr.col(j).tail(m_rows - k - 1).norm();
                    m_colNormsUpdated[j] = m_colNormsDirect[j];
                } else {
                    m_colNormsUpdated[j] *= std::sqrt(temp);
                }
            }
        }
    }

    Index maxIndex(const std::vector<RealScalar>& norms, Index start = 0, Index end = -1) const {
        if (end == -1) end = norms.size();
        return std::distance(norms.begin(), std::max_element(norms.begin() + start, norms.begin() + end));
    }
};

#endif //QR_REDUCED_H
*/
