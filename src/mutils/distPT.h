
// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2023
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2023.08.08

#pragma once

// Compute the distance between a point and a solid triangle in nD.
// 
// The triangle has vertices <V[0],V[1],V[2]>. A triangle point is
// X = sum_{i=0}^2 b[i] * V[i], where 0 <= b[i] <= 1 for all i and
// sum_{i=0}^2 b[i] = 1.
// 
// The input point is stored in closest[0]. The closest point on the triangle
// is stored in closest[1] with barycentric coordinates (b[0],b[1],b[2]).
// 
// For a description of the algebraic details of the quadratic minimization
// approach used by operator(), see
//   https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
// Although the document describes the 3D case, the construction applies in
// general dimensions N. The UseConjugateGradient function uses conjugate
// gradient minimization.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <Eigen/Core>

namespace ADU {

    template <typename T>
    class DistPT
    {
    public:
        struct Result
        {
            Result()
                :
                distance(static_cast<T>(0)),
                sqrDistance(static_cast<T>(0)),
                barycentric{ static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) },
                closest{ Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>::Zero() }
            {
            }

            T distance, sqrDistance;
            std::array<T, 3> barycentric;
            std::array<Eigen::Vector3<T>, 2> closest;
        };

        // This query is exact when using arbitrary-precision arithmetic. It
        // can be used also for floating-point arithmetic, but rounding errors
        // can sometimes lead to inaccurate result. For floating-point,
        // consider UseConjugateGradient(...) which is more robust.
        Result operator()(Eigen::Vector3<T> const& point,
            Eigen::Vector3<T> const& t0,
            Eigen::Vector3<T> const& t1,
            Eigen::Vector3<T> const& t2)
        {
            // The member result.sqrDistance is set in each block of the
            // nested if-then-else statements. The remaining members are all
            // set at the end of the function.
            Result result{};

            T const zero = static_cast<T>(0);
            T const one = static_cast<T>(1);
            T const two = static_cast<T>(2);
            Eigen::Vector3<T> diff = t0 - point;
            Eigen::Vector3<T> edge0 = t1 - t0;
            Eigen::Vector3<T> edge1 = t2 - t0;
            T a00 = edge0.dot(edge0);
            T a01 = edge0.dot(edge1);
            T a11 = edge1.dot(edge1);
            T b0 = diff.dot(edge0);
            T b1 = diff.dot(edge1);
            T det = std::max(a00 * a11 - a01 * a01, zero);
            T s = a01 * b1 - a11 * b0;
            T t = a01 * b0 - a00 * b1;

            if (s + t <= det)
            {
                if (s < zero)
                {
                    if (t < zero)  // region 4
                    {
                        if (b0 < zero)
                        {
                            t = zero;
                            if (-b0 >= a00)
                            {
                                s = one;
                            }
                            else
                            {
                                s = -b0 / a00;
                            }
                        }
                        else
                        {
                            s = zero;
                            if (b1 >= zero)
                            {
                                t = zero;
                            }
                            else if (-b1 >= a11)
                            {
                                t = one;
                            }
                            else
                            {
                                t = -b1 / a11;
                            }
                        }
                    }
                    else  // region 3
                    {
                        s = zero;
                        if (b1 >= zero)
                        {
                            t = zero;
                        }
                        else if (-b1 >= a11)
                        {
                            t = one;
                        }
                        else
                        {
                            t = -b1 / a11;
                        }
                    }
                }
                else if (t < zero)  // region 5
                {
                    t = zero;
                    if (b0 >= zero)
                    {
                        s = zero;
                    }
                    else if (-b0 >= a00)
                    {
                        s = one;
                    }
                    else
                    {
                        s = -b0 / a00;
                    }
                }
                else  // region 0
                {
                    // minimum at interior point
                    s /= det;
                    t /= det;
                }
            }
            else
            {
                T tmp0{}, tmp1{}, numer{}, denom{};

                if (s < zero)  // region 2
                {
                    tmp0 = a01 + b0;
                    tmp1 = a11 + b1;
                    if (tmp1 > tmp0)
                    {
                        numer = tmp1 - tmp0;
                        denom = a00 - two * a01 + a11;
                        if (numer >= denom)
                        {
                            s = one;
                            t = zero;
                        }
                        else
                        {
                            s = numer / denom;
                            t = one - s;
                        }
                    }
                    else
                    {
                        s = zero;
                        if (tmp1 <= zero)
                        {
                            t = one;
                        }
                        else if (b1 >= zero)
                        {
                            t = zero;
                        }
                        else
                        {
                            t = -b1 / a11;
                        }
                    }
                }
                else if (t < zero)  // region 6
                {
                    tmp0 = a01 + b1;
                    tmp1 = a00 + b0;
                    if (tmp1 > tmp0)
                    {
                        numer = tmp1 - tmp0;
                        denom = a00 - two * a01 + a11;
                        if (numer >= denom)
                        {
                            t = one;
                            s = zero;
                        }
                        else
                        {
                            t = numer / denom;
                            s = one - t;
                        }
                    }
                    else
                    {
                        t = zero;
                        if (tmp1 <= zero)
                        {
                            s = one;
                        }
                        else if (b0 >= zero)
                        {
                            s = zero;
                        }
                        else
                        {
                            s = -b0 / a00;
                        }
                    }
                }
                else  // region 1
                {
                    numer = a11 + b1 - a01 - b0;
                    if (numer <= zero)
                    {
                        s = zero;
                        t = one;
                    }
                    else
                    {
                        denom = a00 - two * a01 + a11;
                        if (numer >= denom)
                        {
                            s = one;
                            t = zero;
                        }
                        else
                        {
                            s = numer / denom;
                            t = one - s;
                        }
                    }
                }
            }

            result.closest[0] = point;
            result.closest[1] = t0 + s * edge0 + t * edge1;
            diff = result.closest[0] - result.closest[1];
            result.sqrDistance = diff.dot(diff);
            result.distance = std::sqrt(result.sqrDistance);
            result.barycentric[0] = one - s - t;
            result.barycentric[1] = s;
            result.barycentric[2] = t;
            return result;
        }

        // TODO better cg
    };
}