"""
Author: npgh2009
Time-dependent potential
"""

import numpy as np

class TimeDependentPotential(object):

	def __init__(self, x, pot1, pot2, T0, dt, N_steps):

		self.x, self.pot1, self.pot2 = map(np.asarray, (x, pot1, pot2))
		N = self.x.size
		assert self.x.shape == (N,)
		assert self.pot1.shape == (N,)
		assert self.pot2.shape == (N,)
		self.N = N

		self.T0 = T0
		self.dt = dt
		self.N_steps = N_steps

		self.t = 0
		self.potential = self.pot1.copy()

	def time_step(self):
		self.t += self.N_steps * self.dt
		if self.t < self.T0:
			self.potential = (1 - self.t/self.T0) * self.pot1 + (self.t/self.T0) * self.pot2
		else:
			self.potential = self.pot2.copy()

		