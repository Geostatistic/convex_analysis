import numpy as np
from scipy import sparse as sp
from scipy import linalg as la
from matplotlib import cm
from matplotlib import pyplot as plt

# Line search strong Wolfe with conditions ---------------------------------------------------------
def line_search_wolfe( x, a, p, f, g, c1, c2, m ) :
  alpha_min = 0
  alpha = a
  alpha_max = np.inf

  fx = f( x )
  gx = g( x )
  hx = gx.dot( p )

  z = x + alpha * p
  fz = f( z )
  gz = g( z )
  hz = gz.dot( p ) 

  W1 = fx + alpha * c1 * hx
  W2 = c2 * hx
  i = 0
  while ( ( fz > W1 or hz <= W2 ) and i < m ) :
    if fz > W1:
      alpha_max = alpha
      alpha = 0.5 * ( alpha_min + alpha_max )

    elif hz < W2 :
      if alpha_max >= np.inf :
        alpha = 2 * alpha
      else :
        alpha_min = alpha
        alpha = 0.5 * ( alpha_min + alpha_max )

    z = x + alpha * p
    fz = f( z )
    gz = g( z )
    hz = gz.dot( p )
    i = i + 1

  chk = fz > W1 or hz < W2

  return ( alpha, i, chk )


# Line search strong Wolfe with conditions ---------------------------------------------------------
# Adapted to satisfy convergence conditions of inexact Newton
def line_search_wolfe_2( x, a, p, f, g, h, c1, c2, m ) :
	alpha_min = 0
	alpha = a
	alpha_max = np.inf

	fx = f( x )
	gx = g( x )
	hx = gx.dot( p )

	z = x + alpha * p
	fz = f( z )
	gz = g( z )
	hz = gz.dot( p ) 

	W1 = fx + alpha * c1 * hx
	W2 = c2 * hx
	i = 0
	while ( ( fz > W1 or hz <= W2 ) and i < m ) :
	  if fz > W1:
	    alpha_max = alpha
	    alpha = 0.5 * ( alpha_min + alpha_max )

	  elif hz < W2 :
	    if alpha_max >= np.inf :
	      alpha = 2 * alpha
	    else :
	      alpha_min = alpha
	      alpha = 0.5 * ( alpha_min + alpha_max )

	  z = x + alpha * p
	  fz = f( z )
	  gz = g( z )

	  q = alpha * p
	  # Change to satisfy the condition (7.3) de Nocedal y Wright
	  hz = ( h( z ).dot( q ) + gz ).dot( q ) / np.abs( alpha )
	  i = i + 1

	chk = fz > W1 or hz < W2

	return ( alpha, i, chk )

# Line search Newton CG ----------------------------------------------------------------------------
def ls_newton_cg( x, J, dJ, d2J, N, M, c1, c2, lsi, lsi_sel, err ) :

	F = []
	G = []
	ng = 2 * err
	k = 0

	while k < N and ng > err:
	  alpha = 0
	  z = 0
	  g = dJ( x )
	  ng = la.norm( g )

	  F.append( J( x ) )
	  G.append( ng )

	  r = g
	  B = d2J( x )
	  d = -r
	  e = np.min( [ 0.5, np.sqrt( la.norm( r ) ) ] ) * la.norm( r )

	  j = 0
	  while j < M :
	    kappa = d.T.dot( B.dot( d ) )
	    if kappa <= 0 :
	      if j == 0 :
	        p = -g
	        break
	      else :
	        p = z
	        break

	    alpha = r.dot( r ) / kappa
	    z = z + alpha * d
	    r0 = r
	    r = r + alpha * B.dot( d )
	    if la.norm( r ) < e:
	      p = z
	      break
	    e = np.min( [ 0.5, np.sqrt( la.norm( g ) ) ] ) * la.norm( g )

	    beta = r.dot( r ) / r0.dot( r0 )
	    d = -r + beta * d
	    j = j + 1

	  # Selection of the line search method
	  if lsi_sel == 1:
	    [ alpha, i, chk ] = line_search_wolfe( x, alpha, p, J, dJ, c1, c2, lsi )
	  elif lsi_sel == 2 :
	    [ alpha, i, chk ] = line_search_wolfe_2( x, alpha, p, J, dJ, d2J, c1, c2, lsi )
	  
	  x = x + alpha * p
	  k = k + 1
  
	return [ x, g, F, G, k ]

# Problem definition -------------------------------------------------------------------------------
m = 1000
n = 5
A = np.random.normal( 10, 5, ( m, n ) )
b = np.random.normal( 0, 5, m )

def J( x ) :
  global A, b, alpha, D
  Jx = 0.5 * la.norm( A.dot( x ) - b )**2
  return Jx

K = A.T.dot( A )
c = A.T.dot( b )

def dJ( x ) :
  global K, c, alpha, D
  g = K.dot( x ) - c   
  return g

def d2J( x ) :
  global K
  return K

N = 5000
M = 5
err = 1e-10

c1 = 0.00001
c2 = 0.01
lsi = 10
lsi_sel = 2
x = np.random.normal( 0, 2, n )

[ x, g, F, G, k ] = ls_newton_cg( x, J, dJ, d2J, N, M, c1, c2, lsi, lsi_sel, err )
print( k )
print( x )
print( F[-1] )
print( G[-1] )

# Plotting results ---------------------------------------------------------------------------------
plt.xlabel( 'Iterations' )
plt.ylabel( 'Objective function' )
plt.title( 'Value of the objective function' )
plt.plot( F )

plt.xlabel( 'Iterations' )
plt.ylabel( 'Gradient norm' )
plt.plot( G )

# Checking results with least squares function of numpy --------------------------------------------
xs = np.linalg.lstsq( A, b, rcond = None )
print( la.norm( x - xs[0] ) )