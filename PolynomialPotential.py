"""
Author: npgh2009
Polynomial potential implementation
"""

import numpy as np
from numpy.polynomial.polynomial import polyval

class PolynomialPotential(object):

	def __init__(self, x, coeff, lambd = 1):
		"""
		Parameters
		-----------
		x : numpy.linspace array of length N
		coeff : coefficient for polynomial, in ascending power
		lambd : strength of potential, default to 1
		"""

		# Validation of array inputs
		self.x = np.asarray(x)
		N = self.x.size
		assert self.x.shape == (N,)
		self.N = N

		# Set internal parameters
		self.coeff = numpy.asarray(coeff)
		self.lambd = lambd

		# Compute potential
		self.potential = polyval(x, lambd*self.coeff)