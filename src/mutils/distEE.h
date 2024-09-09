#pragma once
// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2023
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2023.08.08

#pragma once

// Compute the closest points for two segments in nD.
// 
// The segments are P[0] + s[0] * (P[1] - P[0]) for 0 <= s[0] <= 1 and
// Q[0] + s[1] * (Q[1] - Q[0]) for 0 <= s[1] <= 1. The D[i] are not required
// to be unit length.
// 
// The closest point on segment[i] is stored in closest[i] with parameter[i]
// storing s[i]. When there are infinitely many choices for the pair of
// closest points, only one of them is returned.
// 
// The algorithm is robust even for nearly parallel segments. Effectively, it
// uses a conjugate gradient search for the minimum of the squared distance
// function, which avoids the numerical problems introduced by divisions in
// the case the minimum is located at an interior point of the domain. See the
// document
//   https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf
// for details.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <Eigen/Core>


namespace ADU {


    template <typename T>
    class DistEE
    {
    public:
        struct Result
        {
            Result()
                :
                distance(static_cast<T>(0)),
                sqrDistance(static_cast<T>(0)),
                parameter{ static_cast<T>(0), static_cast<T>(0) },
                closest{ Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>::Zero() }
            {
            }

            T distance, sqrDistance;
            std::array<T, 2> parameter;
            std::array<Eigen::Vector3<T>, 2> closest;
        };

        // These two functions are exact for computing Result::sqrDistance
        // when T is a rational type.
        Result operator()(Eigen::Vector3<T> const& P0, Eigen::Vector3<T> const& P1,
            Eigen::Vector3<T> const& Q0, Eigen::Vector3<T> const& Q1)
        {
            Eigen::Vector3<T> P1mP0 = P1 - P0;
            Eigen::Vector3<T> Q1mQ0 = Q1 - Q0;
            Eigen::Vector3<T> P0mQ0 = P0 - Q0;
            T a = P1mP0.dot(P1mP0);
            T b = P1mP0.dot(Q1mQ0);
            T c = Q1mQ0.dot(Q1mQ0);
            T d = P1mP0.dot(P0mQ0);
            T e = Q1mQ0.dot(P0mQ0);

            T det = a * c - b * b;
            T s, t, nd, bmd, bte, ctd, bpe, ate, btd;

            T const zero = static_cast<T>(0);
            T const one = static_cast<T>(1);
            if (det > zero)
            {
                bte = b * e;
                ctd = c * d;
                if (bte <= ctd)  // s <= 0
                {
                    s = zero;
                    if (e <= zero)  // t <= 0
                    {
                        // region 6
                        t = zero;
                        nd = -d;
                        if (nd >= a)
                        {
                            s = one;
                        }
                        else if (nd > zero)
                        {
                            s = nd / a;
                        }
                        // else: s is already zero
                    }
                    else if (e < c)  // 0 < t < 1
                    {
                        // region 5
                        t = e / c;
                    }
                    else  // t >= 1
                    {
                        // region 4
                        t = one;
                        bmd = b - d;
                        if (bmd >= a)
                        {
                            s = one;
                        }
                        else if (bmd > zero)
                        {
                            s = bmd / a;
                        }
                        // else:  s is already zero
                    }
                }
                else  // s > 0
                {
                    s = bte - ctd;
                    if (s >= det)  // s >= 1
                    {
                        // s = 1
                        s = one;
                        bpe = b + e;
                        if (bpe <= zero)  // t <= 0
                        {
                            // region 8
                            t = zero;
                            nd = -d;
                            if (nd <= zero)
                            {
                                s = zero;
                            }
                            else if (nd < a)
                            {
                                s = nd / a;
                            }
                            // else: s is already one
                        }
                        else if (bpe < c)  // 0 < t < 1
                        {
                            // region 1
                            t = bpe / c;
                        }
                        else  // t >= 1
                        {
                            // region 2
                            t = one;
                            bmd = b - d;
                            if (bmd <= zero)
                            {
                                s = zero;
                            }
                            else if (bmd < a)
                            {
                                s = bmd / a;
                            }
                            // else:  s is already one
                        }
                    }
                    else  // 0 < s < 1
                    {
                        ate = a * e;
                        btd = b * d;
                        if (ate <= btd)  // t <= 0
                        {
                            // region 7
                            t = zero;
                            nd = -d;
                            if (nd <= zero)
                            {
                                s = zero;
                            }
                            else if (nd >= a)
                            {
                                s = one;
                            }
                            else
                            {
                                s = nd / a;
                            }
                        }
                        else  // t > 0
                        {
                            t = ate - btd;
                            if (t >= det)  // t >= 1
                            {
                                // region 3
                                t = one;
                                bmd = b - d;
                                if (bmd <= zero)
                                {
                                    s = zero;
                                }
                                else if (bmd >= a)
                                {
                                    s = one;
                                }
                                else
                                {
                                    s = bmd / a;
                                }
                            }
                            else  // 0 < t < 1
                            {
                                // region 0
                                s /= det;
                                t /= det;
                            }
                        }
                    }
                }
            }
            else
            {
                // The segments are parallel. The quadratic factors to
                //   R(s,t) = a*(s-(b/a)*t)^2 + 2*d*(s - (b/a)*t) + f
                // where a*c = b^2, e = b*d/a, f = |P0-Q0|^2, and b is not
                // zero. R is constant along lines of the form s-(b/a)*t = k
                // and its occurs on the line a*s - b*t + d = 0. This line
                // must intersect both the s-axis and the t-axis because 'a'
                // and 'b' are not zero. Because of parallelism, the line is
                // also represented by -b*s + c*t - e = 0.
                //
                // The code determines an edge of the domain [0,1]^2 that
                // intersects the minimum line, or if none of the edges
                // intersect, it determines the closest corner to the minimum
                // line. The conditionals are designed to test first for
                // intersection with the t-axis (s = 0) using
                // -b*s + c*t - e = 0 and then with the s-axis (t = 0) using
                // a*s - b*t + d = 0.

                // When s = 0, solve c*t - e = 0 (t = e/c).
                if (e <= zero)  // t <= 0
                {
                    // Now solve a*s - b*t + d = 0 for t = 0 (s = -d/a).
                    t = zero;
                    nd = -d;
                    if (nd <= zero)  // s <= 0
                    {
                        // region 6
                        s = zero;
                    }
                    else if (nd >= a)  // s >= 1
                    {
                        // region 8
                        s = one;
                    }
                    else  // 0 < s < 1
                    {
                        // region 7
                        s = nd / a;
                    }
                }
                else if (e >= c)  // t >= 1
                {
                    // Now solve a*s - b*t + d = 0 for t = 1 (s = (b-d)/a).
                    t = one;
                    bmd = b - d;
                    if (bmd <= zero)  // s <= 0
                    {
                        // region 4
                        s = zero;
                    }
                    else if (bmd >= a)  // s >= 1
                    {
                        // region 2
                        s = one;
                    }
                    else  // 0 < s < 1
                    {
                        // region 3
                        s = bmd / a;
                    }
                }
                else  // 0 < t < 1
                {
                    // The point (0,e/c) is on the line and domain, so we have
                    // one point at which R is a minimum.
                    s = zero;
                    t = e / c;
                }
            }

            Result result{};
            result.parameter[0] = s;
            result.parameter[1] = t;
            result.closest[0] = P0 + s * P1mP0;
            result.closest[1] = Q0 + t * Q1mQ0;
            Eigen::Vector3<T> diff = result.closest[0] - result.closest[1];
            result.sqrDistance = diff.dot(diff);
            result.distance = std::sqrt(result.sqrDistance);
            return result;
        }
    };

}
