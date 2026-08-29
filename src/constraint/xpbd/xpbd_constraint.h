#pragma once

/// XPBD constraint base + elastic constraint classes (revived from
/// archive/xpbd/XPBD_constraints.h, Macklin et al. 2016 XPBD formulas).
/// Math helpers live in constraint/xpbd_utils.h (shared with the ADMM side).
/// The old XPBDSolver that depended on ADMM internals stays archived.

#include "mutils/common_types.h"
#include "constraint/xpbd_utils.h"

#include <memory>
#include <vector>

class Constraint;

/// Base class for XPBD constraints.
class XPBDConstraint {
public:
    /** indices of the linked bodies */
    std::vector<unsigned int> m_bodies;
    int type = 0; // 0 tri, 1 bending, 2 tet

    ADU::Real m_lambda{};

    ADU::Real stiffness = 1.0;

    explicit XPBDConstraint(const unsigned int numberOfBodies) {
        m_bodies.resize(numberOfBodies);
    }

    unsigned int numberOfBodies() const { return static_cast<unsigned int>(m_bodies.size()); }
    virtual ~XPBDConstraint() = default;

    virtual bool initConstraintBeforeProjection() { return true; };
    virtual bool updateConstraint() { return true; };
    virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta,
        const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt) { return true; };
    virtual bool solveVelocityConstraint(const unsigned int iter) { return true; };
};

/// 2D FEM stretch constraint (cloth), XPBD compliant formulation.
class FEMTriangleConstraint : public XPBDConstraint {
public:
    ADU::Real m_area;
    ADU::Matf_22 m_invRestMat;
    ADU::Real m_xxStiffness;
    ADU::Real m_xyStiffness;
    ADU::Real m_yyStiffness;
    ADU::Real m_xyPoissonRatio;
    ADU::Real m_yxPoissonRatio;

    FEMTriangleConstraint() : XPBDConstraint(3) { type = 0; }

    virtual bool initConstraint(const ADU::Matf_X3& verts, const unsigned int particle1, const unsigned int particle2,
        const unsigned int particle3, const ADU::Real xxStiffness, const ADU::Real yyStiffness, const ADU::Real xyStiffness,
        const ADU::Real xyPoissonRatio, const ADU::Real yxPoissonRatio);
    virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
};

/// Bending constraint (quadratic mean curvature), initialized from an ADU
/// BendingConstraint (rest-shape adjacency).
class XPBDBendingConstraint : public XPBDConstraint {
public:
    ADU::Vecf_3 rest_mean_curvature{};
    ADU::Real rest_mean_curvature_norm{};
    std::vector<ADU::Real> coeffs; // middle vert, ring verts

    XPBDBendingConstraint() : XPBDConstraint(0) { type = 1; }

    virtual bool initConstraint(const std::shared_ptr<Constraint>& bc);
    virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
};

/// 3D FEM strain constraint (soft body), XPBD compliant formulation.
class XPBD_FEMTetConstraint : public XPBDConstraint {
public:
    ADU::Real m_stiffness;
    ADU::Real m_poissonRatio;
    ADU::Real m_volume;
    ADU::Matf_33 m_invRestMat;

    XPBD_FEMTetConstraint() : XPBDConstraint(4) { type = 2; }

    virtual bool initConstraint(const ADU::Matf_X3& verts, const unsigned int particle1, const unsigned int particle2,
        const unsigned int particle3, const unsigned int particle4,
        const ADU::Real stiffness, const ADU::Real poissonRatio);
    virtual bool solvePositionConstraint(ADU::Matf_X3& verts, ADU::Matf_X3& beta, const ADU::Vecf_X& inv_mass, const unsigned int iter, ADU::Real dt);
};
