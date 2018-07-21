"""
Author: npgh2009
Polynomial potential implementation
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt

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
		self.coeff = np.asarray(coeff)
		self.lambd = lambd

		# Compute potential
		self.potential = polyval(x, lambd*self.coeff)

def main():
	x = np.linspace(-3,3,200)
	polypot = PolynomialPotential(x, [0, -5, -7, 0.5, 1], lambd = 1)
	# plt.xlim(-4,4)
	# plt.ylim(-22,22)
	plt.plot(x, polypot.potential, label = 'potential')
	#plt.legend()
	plt.show()

if __name__ == "__main__":
	main()