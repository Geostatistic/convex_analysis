#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pedro Guarderas
"""

# Numerical python library, importing all
import numpy as np

# Scientific computing python library, importing sparse and linalg sections
from scipy import linalg

def line_searh_wolfe( x, a, p, f, g, c1, c2, m ) :
  alpha_min = 1e-15
  alpha = a
  alpha_max = 1e9
    
  fx = f( x )
  gx = g( x )
  
  z = x + alpha * p
  fz = f( z )
  gz = g( z )
  
  hx = gx.dot( p )
  hz = gz.dot( p ) 
  
  W1 = fx + alpha * c1 * hx
  W2 = c2 * hx
  i = 0
  while ( ( fz > W1 or hz < W2 ) and i < m ) :
    if ( fz > W1 ) :
      alpha_max = alpha
      alpha = 0.5 * ( alpha_min + alpha_max )
    elif ( hz < W2 ) :
      if ( alpha_max >= 1e9 ) :
        alpha = 2 * alpha
      else :
        alpha_min = alpha
        alpha = 0.5 * ( alpha_min + alpha_max )
    
    z = x + alpha * p
    fz = f( z )
    gz = g( z )
    hz = gz.dot( p ) 
    W1 = fx + alpha * c1 * hx
    i = i + 1
    
  chk = fz > W1 or hz < W2
  
  return ( alpha, i, chk )

# BFGS optimization method -------------------------------------------------------------------------
def bfgs_opt( f, g, B, x0, n, m, eps, alpha, c1, c2 ) :
  
  # Intialization
  e = 2 * eps
  
  x = x0
  
  fx0 = f( x0 )
  fx = fx0
  
  gx0 = g( x0 )  
  gx = gx0
  
  # Hessian inverse approximation
  H = B
  
  k = 0
  while ( e > eps and k < n ) :
    p = -H.dot( gx )
    
    # Line search    
    ls = line_searh_wolfe( x, alpha, p, f, g, c1, c2, m )
    
    alpha = ls[0]
    s = alpha * p
    
    # Updating values
    x0 = x
    x = x + s
    
    fx0 = fx
    fx = f( x )
    
    gx0 = gx
    gx = g( x )
    y = gx - gx0
    
    # BFGS algorithm
    rho = y.dot( s )
    u = np.identity( x.size ) - np.outer( y, s ) / rho
    H = u.T.dot( H.dot( u ) ) + np.outer( s, s ) / rho
    e = linalg.norm( y )  
    k = k + 1
    
  return ( x, fx, gx, p, H, e, k )

# Rosenbrock example -------------------------------------------------------------------------------
# Rosenbrock function
def f( x ) :
  f = 100 * ( x[1] - x[0]**2 )**2 + ( 1 - x[0] )**2
  return f

# Rosenbrock gradient
def g( x ) :
  gf = np.array( [ 400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * ( x[1] - x[0]**2 ) ] )
  gf = gf.T
  return gf

n = 1000
m = 200
eps = 1e-12
c1 = 0.001
c2 = 0.01
alpha = 0.1
x0 = np.array( [ 1.2, 1.2 ] )
B = np.identity( 2 )
S = bfgs_opt( f, g, B, x0, n, m, eps, alpha, c1, c2 )
[ print( s ) for s in S ]

n = 1000
m = 200
eps = 1e-12
c1 = 0.001
c2 = 0.01
alpha = 0.1
x0 = np.array( [ -1.2, 1.1 ] )
B = np.identity( 2 )
S = bfgs_opt( f, g, B, x0, n, m, eps, alpha, c1, c2 )
[ print( s ) for s in S ]
