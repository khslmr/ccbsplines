#Copyright (C) 2020 Karl Haislmaier

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from numba import njit, vectorize, jitclass
from numba import float64, int64, boolean, typeof, types
from numba.jitclass.base import register_class_type, ClassBuilder


def CubicBSpline(xi, yi, axis=0, jit=True):
	"""A fast, bare bones routine (optimized with numba) for Cubic Cardinal 
	B Spline Interpolation.
	
	Parameters
	----------
	xi : array, shape (n,)
	    1-D array of the independent values.  Values must be strictly increasing 
	    and uniformly spaced.
	
	yi : array, shape (n, ...)
	    N-D array of the dependent values.  The interpolation takes place along the
	    first axis.
	
	Methods
	-------
	``interp(x)``
	    The interpolation points, x, may be either 1-D with size n, or N-D with 
	    with the same shape as yi.
	
	Notes
	-----
	This is an implementation of the uniform grid cubic spline interpolation 
	algorithm of Habermann & Kindermann (2007):
		http://www.springerlink.com/index/10.1007/s10614-007-9092-4
	
	To make this work, I adapted (with minor modifications) the "Thomas Solver," 
	a TDMA solver, implemented in numba here: 
	    https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469
	and here
	    https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
	and the latter is also under a GPL license.  Given these source code licenses, 
	the most permissible copyright license I can provide this module under is GPL3.
	"""
	if not hasattr(CubicBSpline, 'jitted_versions'):   #a rare use of function attrs
		CubicBSpline.jitted_versions = {}
	
	ndim = yi.ndim
	if axis < 0:
		axis += ndim
	
	if axis > 0:
		ax_reorder = (axis,) + tuple([a for a in range(ndim) if not a==axis])
		ax_restore = tuple([a for a in range(1, axis+1)]) + (0,) + \
					 tuple([a for a in range(axis+1, ndim)])
	else:
		ax_reorder = None
		ax_restore = None
	
	ax_expand = tuple([i for i in range(1, yi.ndim)])
	
	key = (ndim, axis, jit)
	if not key in CubicBSpline.jitted_versions:
		if ax_reorder is None:
			tup_type = typeof( tuple(np.arange(ndim)) )
		else:
			tup_type = typeof(ax_reorder)
		
		slc = [slice(None)] * ndim
		spec = [('xi', float64[:]), 
		        ('c', float64[slc]),
		        ('ordered', boolean),
		        ('_ax_reorder', tup_type ),
		        ('_ax_restore', tup_type ),
		        ('_ax_expand', typeof(ax_expand) ),]
		if jit:
			cls = register_class_type(_CubicBSpline, spec, types.ClassType, ClassBuilder)
		else:
			cls = _CubicBSpline
		CubicBSpline.jitted_versions[key] = cls
	else:
		cls = CubicBSpline.jitted_versions[key]
	
	return cls(xi, yi, ax_reorder, ax_restore, ax_expand)


class _CubicBSpline(object):
	"""A fast, bare bones routine (optimized with numba) for Cubic Cardinal 
	B Spline Interpolation.
	
	Parameters
	----------
	xi : array, shape (n,)
	    1-D array of the independent values.  Values must be strictly increasing 
	    and uniformly spaced.
	
	yi : array, shape (n, ...)
	    N-D array of the dependent values.  The interpolation takes place along the
	    first axis.
	
	Methods
	-------
	``interp(x)``
	    The interpolation points, x, may be either 1-D with size n, or N-D with 
	    with the same shape as yi.
	
	Notes
	-----
	This is an implementation of the uniform grid cubic spline interpolation 
	algorithm of Habermann & Kindermann (2007):
		http://www.springerlink.com/index/10.1007/s10614-007-9092-4
	
	My code is adapted Joon Ro's `fast-cubic-spline-python` cython implementation 
	(under a GPL3 license) by porting to numba/numpy.  
		https://github.com/joonro/fast-cubic-spline-python
	I also adapted (with minor modifications) a TDMA solver from here 
	    https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469
	and here
	    https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
	and the latter is also under a GPL license.  Given these source code licenses, 
	the most permissible copyright license I can provide this module under is GPL3.
	"""
	
	def __init__(self, xi, yi, ax_reorder, ax_restore, ax_expand):
		self._ax_expand = ax_expand
		
		if ax_reorder is None:
			self.ordered = True
			coeffs = calc_coeffs(yi)
		else:
			self.ordered = False
			self._ax_reorder = ax_reorder
			self._ax_restore = ax_restore
			coeffs = calc_coeffs( yi.transpose(*self._ax_reorder) )
		
		shp = (1,) + coeffs.shape[1:]
		self.c = np.concatenate((coeffs, np.zeros(shp)))
		self.xi = xi
	
	def interp(self, x):
		shp = (x.shape[0],) + self.c.shape[1:]
		y = np.zeros(shp)
		
		h = (self.xi[-1] - self.xi[0]) / float64(self.xi.shape[0]-1)
		aux = (x - self.xi[0]) / h
		k = aux.astype(np.int64) - 1   #if following H&K07, this `k` is their "k-2"
		
		for i in range(4):
			abs_diff = np.abs(aux - k).reshape((x.shape[0],) + self._ax_expand)
			k += 1   #k must be incremented between these lines
			u = cubic_Bspline_kernel(abs_diff)
			y += u * self.c[k]
		
		if self.ordered:
			return y
		else:
			return y.transpose(*self._ax_restore)


@vectorize([float64(float64,)])
def cubic_Bspline_kernel(abs_t):
	if abs_t > 2:
		return 0.
	elif abs_t > 1:
		return (2. - abs_t)**3
	else:
		return 4. - 3. * (2. * abs_t**2 - abs_t**3)

@njit
def calc_coeffs(yi, alpha=0., beta=0.):
	out_shp = (yi.shape[0]+2,) + yi.shape[1:]
	c = np.zeros(out_shp)
	
	c[1] = (yi[0] - alpha / 6.) / 6.
	c[-2] = (yi[-1] - beta / 6.) / 6.
	
	y = yi[1:-1].copy()
	y[0] -= c[1]
	y[-1] -=  c[-2]
	
	a = np.ones(y.shape)   #the TDMAsolver ignores a[-1]
	b = 3. + a
	c[2:-2] = TDMAsolver(a, b, a.copy(), y)
	
	c[0] = alpha / 6. + 2. * c[1] - c[2]
	c[-1] = beta / 6. + 2. * c[-2] - c[-3]
	
	return c

@njit
def TDMAsolver(a, b, c, d):
	"""
	TDMA solver for in Ax = d, where: 
		a = A.diagonal(-1)
		b = A.diagonal(0)
		c = A.diagonal(1)
		A is zero elsewhere
		a, b, c, and d are NumPy arrays
	If any of (a, b, c, d) are greater than 1-D, the solver executes along the 
	first axis.

	This algorithm (a.k.a. the 'Thomas Solver') is known to be stable when A is 
	diagonally dominant.  Refer to 
	    http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
	and 
	    http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

	This implementation is adapted closely from 
	    https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
	and 
	    https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469
	which are under GPL3 licenses."""
	
	nf = d.shape[0]
	for it in range(0, nf-1):
		m = a[it] / b[it]
		b[it+1] -= m * c[it]
		d[it+1] -= m * d[it]
	
	b[-1] = d[-1] / b[-1]
	for il in range(nf-2, -1, -1):
		b[il] = (d[il] - c[il] * b[il+1]) / b[il]
	
	return b

