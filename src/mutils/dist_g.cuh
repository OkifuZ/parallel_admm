#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <type_traits>



template <typename T>
struct Result
{
    T distance{};
    T sqrDistance{};
    T barycentric[3]{};
    Eigen::Vector3<T> closest[2]{};
};

// This query is exact when using arbitrary-precision arithmetic. It
// can be used also for floating-point arithmetic, but rounding errors
// can sometimes lead to inaccurate result. For floating-point,
// consider UseConjugateGradient(...) which is more robust.
template <typename T>
__inline__ __device__ Result<T> dcdPT(Eigen::Vector3<T> const& point,
    Eigen::Vector3<T> const& t0,
    Eigen::Vector3<T> const& t1,
    Eigen::Vector3<T> const& t2)
{
    // The member result.sqrDistance is set in each block of the
    // nested if-then-else statements. The remaining members are all
    // set at the end of the function.
    Result<T> result{};
    bool valid = false;

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
            valid = true;
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

    if (!valid) {
        result.distance = -10;
        return result;
    }
    result.closest[0] = point;
    result.closest[1] = t0 + s * edge0 + t * edge1;
    diff = result.closest[0] - result.closest[1];
    result.sqrDistance = diff.dot(diff);
    if constexpr (std::is_floating_point_v<T>){
        result.distance = sqrtf(result.sqrDistance);
    }
    else { 
        result.distance = sqrt(result.sqrDistance);
    }
    result.barycentric[0] = one - s - t;
    result.barycentric[1] = s;
    result.barycentric[2] = t;

    
    return result;
}

// These two functions are exact for computing Result::sqrDistance
// when T is a rational type.
template <typename T>
__inline__ __device__ Result<T> dcdEE(Eigen::Vector3<T> const& P0, Eigen::Vector3<T> const& P1,
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

    bool valid = false;

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
                        valid = true;
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
            //valid = true;
        }
    }

    Result<T> result{};
    if (!valid) {
        result.distance = -10;
        return result;
    }
    result.barycentric[0] = s;
    result.barycentric[1] = t;
    result.barycentric[2] = -10;
    result.closest[0] = P0 + s * P1mP0;
    result.closest[1] = Q0 + t * Q1mQ0;
    Eigen::Vector3<T> diff = result.closest[0] - result.closest[1];
    result.sqrDistance = diff.dot(diff);
    if constexpr (std::is_floating_point_v<T>){
        result.distance = sqrtf(result.sqrDistance);
    }
    else { 
        result.distance = sqrt(result.sqrDistance);
    }
    result.distance = std::sqrt(result.sqrDistance);

    /*if (!valid) {
        result.distance = -10;
    }*/
    return result;
}