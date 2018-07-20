"""
Author: npgh2009
Quantum harmonic oscillator implementation
"""

import numpy as np
from math import factorial, sqrt
from numpy.polynomial.hermite import hermval
import matplotlib.pyplot as plt

class HarmonicOscillator(object):

	def __init__(self, x, omega, hbar = 1.0, m = 0.5):
		"""
		Initialize the potential for quantum harmonic oscillator
		Parameters
		------------
		x : numpy.linspace array of length N
		omega : float
		hbar : float, default to 1.0
		m : float, default to 0.5
		"""

		# Validation of array inputs
		self.x = np.asarray(x)
		N = self.x.size
		assert self.x.shape == (N,)
		self.N = N

		# Validate and set internal parameters
		assert hbar > 0
		assert m > 0
		assert omega > 0
		self.hbar = hbar
		self.m = m
		self.omega = omega

        # Set useful parameters
		self.alpha = m * omega / hbar
		self.y = sqrt(self.alpha) * x

        # Compute the potential from array x
		self.potential = (0.5 * m * omega**2) * self.x**2

	def eigenFunction(self, n = 0):
		"""
		Parameters
		--------
		n : int, quantum number of eigenstate
		"""
		if n > 0:
			coeff = [0 for _ in range(n)]
			coeff.append(1)
		else:
			coeff = [1]
		return (self.alpha/np.pi)**(0.25) * (2**n * factorial(n))**(-0.5) * hermval(self.y, coeff) * np.exp(-0.5 * self.y**2)

	def coherenceState(self, v, n = 0):
		"""
		Parameters
		--------
		v : float, velocity of state
		n : int, quantum number of eigenstate
		"""
		if n > 0:
			coeff = [0 for _ in range(n)]
			coeff.append(1)
		else:
			coeff = [1]
		k = self.m*v/self.hbar
		return ((self.alpha/np.pi)**(0.25) * (2**n * factorial(n))**(-0.5) * hermval(self.y, coeff)
				* np.exp(-0.5 * self.y**2 + 1j * k * self.x))

	def squeezeState(self, omegasq, n = 0):
		"""
		Parameters
		--------
		n : int, quantum number of eigenstate
		omegasq : omega of QHO before the squeeze
		"""
		alphasq =  self.m * omegasq / self.hbar
		ysq = sqrt(alphasq) * self.x
		if n > 0:
			coeff = [0 for _ in range(n)]
			coeff.append(1)
		else:
			coeff = [1]
		return (alphasq/np.pi)**(0.25) * (2**n * factorial(n))**(-0.5) * hermval(ysq, coeff) * np.exp(-0.5 * ysq**2)


def main():
	x = np.linspace(-20,20,200)
	qho = HarmonicOscillator(x, omega = 0.05)
	plt.plot(x, qho.potential, label = 'potential')
	plt.plot(x, qho.eigenFunction(), label = 'eigenfn')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()