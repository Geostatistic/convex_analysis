#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pedro Guarderas
"""

s = x - xp
u = y - B.dot( s )

# Symmetric-rank on SR1
B = B - u.dot( u.T ) / u.dot( s )

# BFGS formula
v = B.dot( s )
B = B - v.dot( v.T ) / s.dot( v ) + y.dot( y.T ) / y.dot( s )

# BFGS inverse formula
rho = 1.0 / y.dot( s )
u = ( I - rho * s.dot( y.T ) )
H = u.dot( H.dot( u ) ) + rho * s.dot( s.T )

# Non linear conjugate gradiente method
p = -g + beta * pp

# Linear regression example ------------------------------------------------------------------------
A = np.array( [ [ -2.04, -0.88, -0.05, -0.16, 1.42, 1.02, 0.06, -0.88, 0.89, 0.62 ],
                [ -1.28, 0.32, 1.03, -1.28, -1.28, 1.39, 1.39, -0.04, -0.21, -0.04 ] ] )
A = A.T
K = A.T.dot( A )

y = np.array( [ -0.94, -0.52, -0.05, -0.18, 0.67, 0.59, 0.19, -0.12, -0.17, 0.21 ] )
y = y.T
b = A.T.dot( y )

def f( x ) :
  global A, y
  u = A.dot( x ) - y
  return u.dot( u )

def g( x ) :
  global K, A, y
  return 2.0 * ( K.dot( x ) - A.T.dot( y ) )

n = 10000
m = 20
eps = 1e-11
x0 = np.ones( 2 )
B = np.identity( 2 )
c1 = 0.01
c2 = 0.1
alpha = 0.001
S = bfgs_opt( f, g, B, x0, n, m, eps, alpha, c1, c2 )
[ print( s ) for s in S ]
xs = linalg.solve( K, b )
print( xs )

del A, K, y, b, xs, S
