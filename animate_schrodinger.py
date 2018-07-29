"""
Solve and animate the Schrodinger equation

First presented at http://jakevdp.github.com/blog/2012/09/05/quantum-python/

Author: Jake Vanderplas <vanderplas@astro.washington.edu>
License: BSD
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from schrodinger_time_dependent import Schrodinger
from HarmonicOscillator import HarmonicOscillator
from PolynomialPotential import PolynomialPotential
from TimeDependentPotential import TimeDependentPotential

# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

# specify time steps and duration
dt = 0.01
N_steps = 20
t_max = 120
frames = int(t_max / float(N_steps * dt))

# specify constants
hbar = 1.0   # planck's constant
m = 0.5      # particle mass

# specify range in x coordinate
N = 50
dx = 0.1
nx = int(N/dx)*2 + 1
x = np.linspace(-N, N, nx)

# specify potential
omega = 0.7 #the higher, the sharper is the potential, and eigenfns more localized
qho = HarmonicOscillator(x = x, omega = omega, hbar = hbar, m = m)
qho2 = HarmonicOscillator(x = x, omega = 2, hbar = hbar, m = m)
lambd = 0.1
polypot = PolynomialPotential(x = x, coeff = [0, -5, -7, 0.5, 1], lambd = lambd)
T0 = 50
mixedpot = TimeDependentPotential(x = x, pot1 = qho.potential, pot2 = qho2.potential, T0 = T0, dt = dt, N_steps = N_steps)
V_x = mixedpot.potential

# # specify initial momentum and quantities derived from it
# p0 = np.sqrt(2 * m * 0.2 * V0)
# dp2 = p0 * p0 * 1. / 80
# d = hbar / np.sqrt(2 * dp2)

v0 = 0.2
p0 = m * v0
k0 = p0 / hbar

# specify initial wavefunction
psi_x0 = qho.eigenFunction(n = 0)

# define the Schrodinger object which performs the calculations
S = Schrodinger(x=x,
                psi_x0=psi_x0,
                V_x=V_x,
                hbar=hbar,
                m=m,
                k0=-50)

######################################################################
# Set up plot
fig = plt.figure()

# plotting limits
xlim = (-8, 8)
ymin = -2
ymax = 4

ax1 = fig.add_subplot(111, xlim=xlim,
                            ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')
#V_x_line2, = ax1.plot(x, qho.potential, c='b')
zero_line = ax1.axhline(0, c='gray', lw = 1)
timetext = ax1.text(-1, 4.9, "")

ax1.legend(prop=dict(size=12))
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$|\psi(x)|$')

V_x_line.set_data(S.x, S.V_x)


######################################################################
# Functions to Animate the plot
def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])
    timetext.set_text("")

    return (psi_x_line, V_x_line, timetext)


def animate(i):
    mixedpot.time_step()
    V_x = mixedpot.potential
    S._set_V_x(V_x)
    S.time_step(dt, N_steps)
    psi_x_line.set_data(S.x, 4*abs(S.psi_x))
    V_x_line.set_data(S.x, S.V_x)
    timetext.set_text("t = %.2fs" % S.t)

    return (psi_x_line, V_x_line, timetext)

# call the animator.
# blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)


# uncomment the following line to save the video in mp4 format.  This
# requires either mencoder or ffmpeg to be installed on your system
#anim.save('schrodinger_time_dependent.mp4', fps=15,
#          extra_args=['-vcodec', 'libx264'])

plt.show()
