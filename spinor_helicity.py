"""
Spinor-helicity utilities for symbolic calculations in SymPy.

This module implements:
    - Pauli four-vectors:                            sig, sig_bar
    - Epsilon tensors for SL(2,C) and SU(2):         ueps, leps
    - Conversion spherical to Cartesian coordinates: CartesianCoords(p)
    - Conversion four-momenta to bi-spinors:         MSI(p)
    - Tensor product:                                TensorProd(A,B)
    - Helicity- and spin-spinors:                    HelicitySpinors(p), SpinSpinors(p)
    - Helicity-spinor products:                      HelicityAngleProd(A,B), HelicitySquareProd(A,B)
    - Spin-spinor products:                          SpinAngleProd(A,B), SpinSquareProd(A,B)

Conventions:
    - Metric signature: (+, −, −, −).
    - Four-vectors are given as [E, P, theta, phi] in spherical coordinates
      unless otherwise specified.
    - Helicity basis for spin projections
    - HelicitySpinors(p) returns [<p|, |p>, [p|, |p]]
    - SpinSpinors(p) returns the analogous set of spin-spinors with upper SU(2) indices.
    
Note:
    - Both functions, HelicitySpinors and SpinSpinors, return ALL helicity- or spin-spinors.
      Additionally the spinor product functions are defined such that they pick out the 
      correct helicity- or spin-spinors when given two HelicitySpinors or SpinSpinors objects
      So one doesn't have to define indiviual objects for each helicity- or spin-spinor which 
      is way more convenient.
"""

from sympy import Matrix, I, eye, sin, cos, exp, sqrt
from sympy.physics.matrices import msigma
from sympy import tensor

# Pauli 4-vectors
sig = [eye(2), msigma(1), msigma(2), msigma(3)]
sig_bar = [eye(2), -msigma(1), -msigma(2), -msigma(3)]

# Antisymmetric 2x2 matrix with upper SL(2,C)/ little-group indices, eps^{ab}
ueps = Matrix([
    [0, 1],
    [-1, 0]
])

# Antisymmetric 2x2 matrix with lower SL(2, C)/ little-group indices, eps_{ab}
leps = Matrix([
    [0, -1],
    [1, 0]
])

# Useful functions
def CartesianCoords(p):
    """
    Takes a python list of sympy.Symbol objects corresponding to a 4-momentum in spherical coordinates [E, P, θ, φ].
    Returns a sympy.Matrix object corresponding to the 4-momentum in cartisian coordinates.
    """
    En, P, theta, phi = p[0], p[1], p[2], p[3]
    return Matrix([En, P*sin(theta)*cos(phi), P*sin(theta)*sin(phi), P*cos(theta)])

def MSI(p):
    """
    Minkowski Space Isomorphism R^4 -> C^{2x2}. 
    
    Takes a python list corresponding to 4-vector in cartesian coordinates.
    Returns sympy.Matrix corresponding to the 4-vector's bi-spinor.
    """
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    return Matrix([
        [p0 - p3, -p1 + I*p2],
        [-p1 - I*p2, p0 + p3]
    ])

def TensorProd(A, B):
    """
    Takes two sympy.Matrix objects.
    Returns a sympy.tensor.MutableDenseNDimArray object corresponding to the tensor product of A and B
    """
    m, n = A.shape
    p, q = B.shape
    result = tensor.array.MutableDenseNDimArray.zeros(m, n, p, q)

    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    result[i, j, k, l] = A[i, j] * B[k, l]

    return result

# Definition of helicity-spinors and products
def HelicitySpinors(p):
    """
    Takes a python list of sympy.Symbol objects corresponding to a 4-vector in spherical coords [E, P, theta, phi].
    Returns a python list of sympy.Matrix objects corresponding to the 4-vector's helicity-spinors [abra, aket, sbra, sket]
    """
    En, P, theta, phi = p[0], p[1], p[2], p[3]

    abra = sqrt(2*En) * Matrix([cos(theta/2), sin(theta/2)*exp(-I*phi)])
    aket = sqrt(2*En) * Matrix([-sin(theta/2)*exp(-I*phi), cos(theta/2)])
    sbra = sqrt(2*En) * Matrix([-sin(theta/2)*exp(+I*phi), cos(theta/2)])
    sket = sqrt(2*En) * Matrix([cos(theta/2), sin(theta/2)*exp(+I*phi)])

    return [abra, aket, sbra, sket]

def HelicityAngleProd(A,B):
    """
    Computes the angle helicity-spinor product ⟨AB⟩.
    
    Takes two HelicitySpinors objects.
    Returns a sympy.Matrix object corresponding to the angle helicity-spinor product of the angle helicity-spinors contained in A and B.
    """
    return A[0].T * B[1]

def HelicitySquareProd(A,B):
    """
    Computes the square helicity-spinor product [AB].
    
    Takes two HelicitySpinors objects.
    Returns a sympy.Matrix object corresponding to the square helicity-spinor product of the square helicity-spinors contained in A and B.
    """
    return A[2].T * B[3]

# Definition of spin-spinors and products
def SpinSpinors(p):
    """
    Takes a python list of sympy.Symbol objects corresponding to a 4-vector in spherical coords [E, P, theta, phi].
    Returns python list of sympy.Matrix objects corresponding to the 4-vector's spin-spinors [abra, aket, sbra, sket] with *upper* SU(2) indices.
    """
    En, P, theta, phi = p[0], p[1], p[2], p[3]

    abra = Matrix([
        [sqrt(En - P)*exp(I*phi)*sin(theta/2), sqrt(En + P)*cos(theta/2)], 
        [-sqrt(En - P)*cos(theta/2), sqrt(En + P)*exp(-I*phi)*sin(theta/2)]
    ])
    aket = Matrix([
        [sqrt(En - P)*cos(theta/2), -sqrt(En + P)*exp(-I*phi)*sin(theta/2)], 
        [sqrt(En - P)*exp(I*phi)*sin(theta/2), sqrt(En + P)*cos(theta/2)]
    ])
    sbra = Matrix([
        [-sqrt(En + P)*exp(I*phi)*sin(theta/2), -sqrt(En - P)*cos(theta/2)], 
        [sqrt(En + P)*cos(theta/2), -sqrt(En - P)*exp(-I*phi)*sin(theta/2)]
    ])
    sket = Matrix([
        [sqrt(En + P)*cos(theta/2), -sqrt(En - P)*exp(-I*phi)*sin(theta/2)], 
        [sqrt(En + P)*exp(I*phi)*sin(theta/2), sqrt(En - P)*cos(theta/2)]
    ])

    return [abra, aket, sbra, sket]

def SpinAngleProd(A, B):
    """
    Computes the angle spin-spinor product ⟨AB⟩^{IJ}.
    
    Takes two SpinSpinors objects.
    Returns a sympy.Matrix object corresponding to the angle spin-spinor product of the angle spin-spinors contained in A and B.
    """
    return A[0].T * B[1]

def SpinSquareProd(A, B):
    """
    Computes the square spin-spinor product [AB]^{IJ}.
    
    Takes two SpinSpinors objects.
    Returns a sympy.Matrix object corresponding to the square spin-spinor product of the square spin-spinors contained in A and B.
    """
    return A[2].T * B[3]
