#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pedro Guarderas
"""

# Libraries ----------------------------------------------------------------------------------------
# Numerical python library, importing all
import numpy as np

# Scientific computing python library, importing sparse and linalg sections
from scipy import linalg

# Implementation of the dogleg method --------------------------------------------------------------
def dogleg_method( f, g, B, delta ) :
  
  pu = -( g.T.dot( g ) / ( g.T.dot( B.dot( g ) ) ) ) * g
  pb = -linalg.solve( B, g )
  
  npb = linalg.norm( pb ) 
  
  if npb > delta :
    tau = delta / npb
    p = tau * pb
  else :
    a = linalg.norm( pb - pu )**2
    b = pu.T.dot( pb - pu )
    c = linalg.norm( pu )**2 - delta**2
    tau = ( 2.0 * a - b + np.sqrt( b**2 - 4.0 * a * c ) ) / ( 2.0 * a )
    p = pu + ( tau - 1 ) * ( pb - pu )  
    
  return ( p, tau )
    
def trust_region_method( x0, f, g, h, app_meth, delta, eta, eps, ite ) :
 
  # Intialization
  x = x0
  
  fx0 = f( x0 )
  fx = fx0
  
  gx0 = g( x0 )  
  gx = gx0
  p = gx
  
  Bx0 = h( x0 )
  Bx = Bx0
  
  mx0 = fx
  mx = fx0 + gx.T.dot( p ) + 0.5 * p.T.dot( Bx.dot( p ) )
  rho = 1
   
  e = 2 * eps
  ev = []
  dk = delta
  k = 0
  while k < ite and e > eps :
    
    # Approximation of p
    p, tau = app_meth( fx, gx, Bx, dk )
    
    # Evaluation of rho
    mx0 = mx
    mx = fx + gx.T.dot( p ) + 0.5 * p.T.dot( Bx.dot( p ) )
    
    if mx0 > mx :
        rho = ( fx0 - f( x + p ) ) / ( mx0 - mx )
    else:
        rho = ( fx0 - f( x + p ) ) / 1e-10
    
    # Algorithm for trust region
    if rho < 0.25 :
      dk = 0.25 * dk
    else :
      if rho > 0.75 and linalg.norm( p ) == dk :
        dk = min( 2 * dk, delta )

    if rho > eta :
      x0 = x
      x = x + p
    else :
      x0 = x 

    # Updating values
    fx0 = fx
    fx = f( x )
    
    gx0 = gx
    gx = g( x )
    
    Bx0 = Bx
    Bx = h( x )
        
    e = linalg.norm( gx ) 
    ev.append( e )
    k = k + 1
    
  return ( x, fx, gx, Bx, k, ev )

# Rosenbrock ---------------------------------------------------------------------------------------
# Rosenbrock function
def f( x ) :
  f = 100 * ( x[1] - x[0]**2 )**2 + ( 1 - x[0] )**2
  return f

# Rosenbrock gradient
def g( x ) :
  gf = np.array( [ 400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * ( x[1] - x[0]**2 ) ] )
  gf = gf.T
  return gf

# Rosenbrock Hessian
def h( x ) :
  B = np.zeros( ( 2, 2 ) )
  B[ 0, 0 ] = 2.0 - 400.0 * x[1] + 1200.0 * x[0]**2
  B[ 0, 1 ] = -400.0 * x[0]
  B[ 1, 0 ] = -400.0 * x[0]
  B[ 1, 1 ] = 200.0
  return B

ite = 1000
eps = 1e-12
delta = 100
eta = 1e-20
x0 = np.array( [ 1.2, 1.2 ] )
S = trust_region_method( x0, f, g, h, dogleg_method, delta, eta, eps, ite )
[ print( S[i] ) for i in range( 5 ) ]
  
ite = 1000
eps = 1e-12
delta = 100
eta = 1e-20
x0 = np.array( [ -1.2, 1.1 ] )
S = trust_region_method( x0, f, g, h, dogleg_method, delta, eta, eps, ite )
[ print( S[i] ) for i in range( 5 ) ]
