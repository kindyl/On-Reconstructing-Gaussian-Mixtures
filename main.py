
import sympy
import numpy as np
from math import factorial,comb
from sympy import nsimplify, expand, bell


# used to get expansion coeffs
def g_derivs(n):
    dij = sympy.Symbol('d_{ij}')
    var = sympy.Symbol('sigma^2')
    return [dij * (4*var)**(num-1) * factorial(num) for num in range(1,n+1)]


# used to get expansion coeffs
def complete_bell(n):
    if n==0:
        return 1
    sum = 0
    for k in range(1,n+1):
        sum += bell(n,k, g_derivs(n-k+1))
    return sum


# gets expansion coeffs of the moments
def expansion_coeff(n):
    var = sympy.Symbol('sigma^2')
    sum = 0
    for m in range(n+1):
        sum += factorial(2*m) / factorial(m)**2 * var**m * complete_bell(n-m) / factorial(n-m)
    return sum


# gets leading coeffs of the p_i as functions of sigma^2
def leading_coeffs(B,bii):
    var = sympy.Symbol('sigma^2')
    m = len(bii)
    LT = sympy.zeros(m,1)
    LT[0] = -bii[0] / B[0,0]

    for j in range(1,m):
        sum = 0
        for n in range(j):
            sum += B[j,n]*LT[n]
        LT[j] = (-bii[j] -  sum) / B[j,j]

    LC = [LT[i].coeff(var**(i+1)) for i in range(m)]
    return LC


# gets the mth elementary symmetric polynomial
def elem_symm_poly(m):
    p = sympy.Matrix(sympy.symbols('p1:{}'.format(m+1)))
    M = sympy.zeros(m,m)
    for i in range(m):
        M += p[i]*np.eye(m,k=-i)
    for i in range(m-1):
        M[i,i+1] = i+1 
    return p,nsimplify(M).det() / factorial(m) 

# returns matrix of system of equations from moments of Delta_ij
def moment_system(k):
    kprime = comb(k,2)+1
    dij = sympy.Symbol('d_{ij}')
    B = sympy.zeros(kprime,kprime)
    a = sympy.Matrix(sympy.symbols('a1:{}'.format(kprime+1)))
    bii = sympy.zeros(kprime,1)

    for row in range(kprime):
        p = expand(expansion_coeff(row+1))
        # myprint(p)
        q = sympy.Poly(p,dij)
        bii[row] = k**2 * q.coeffs()[-1]
        for col in range(kprime):
            B[row,col] = 2*p.coeff(dij**(col+1))

    B = nsimplify(B)
    bii = nsimplify(bii)
    rhs = a - bii # used lhs and sol to verify k=3 case
    return B,rhs