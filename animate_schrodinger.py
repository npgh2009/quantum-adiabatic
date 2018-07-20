"""
Solve and animate the Schrodinger equation

First presented at http://jakevdp.github.com/blog/2012/09/05/quantum-python/

Author: Jake Vanderplas <vanderplas@astro.washington.edu>
License: BSD
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from schrodinger import Schrodinger
from HarmonicOscillator import HarmonicOscillator


######################################################################
# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))


def gauss_k(k, a, x0, k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi)) ** 0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))


######################################################################
# Utility functions for running the animation
def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y


def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

######################################################################
# Create the animation

# specify time steps and duration
dt = 0.01
N_steps = 50
t_max = 120
frames = int(t_max / float(N_steps * dt))

# specify constants
hbar = 1.0   # planck's constant
m = 0.5      # particle mass

# specify range in x coordinate
N = 20
dx = 0.1
nx = int(N/dx)*2 + 1
x = np.linspace(-N, N, nx)

# specify potential
omega = 0.1
qho = HarmonicOscillator(x = x, omega = omega, hbar = hbar, m = m)
V_x = qho.potential

# # specify initial momentum and quantities derived from it
# p0 = np.sqrt(2 * m * 0.2 * V0)
# dp2 = p0 * p0 * 1. / 80
# d = hbar / np.sqrt(2 * dp2)

v0 = 0.2
p0 = m * v0
k0 = p0 / hbar

# specify initial wavefunction
#psi_x0 = qho.eigenFunction(n = 0)
psi_x0 = qho.coherenceState(v0, n = 0)

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
xlim = (-N, N)
#klim = (-5, 5)

# top axes show the x-space data
ymin = 0
ymax = 10*hbar*omega
ax1 = fig.add_subplot(111, xlim=xlim,
                            ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')

title = ax1.set_title("")
ax1.legend(prop=dict(size=12))
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$|\psi(x)|$')

# bottom axes show the k-space data
# ymin = abs(S.psi_k).min()
# ymax = abs(S.psi_k).max()
# ax2 = fig.add_subplot(212, xlim=klim,
#                       ylim=(ymin - 0.2 * (ymax - ymin),
#                             ymax + 0.2 * (ymax - ymin)))
# psi_k_line, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')

# p0_line1 = ax2.axvline(-p0 / hbar, c='k', ls=':', label=r'$\pm p_0$')
# p0_line2 = ax2.axvline(p0 / hbar, c='k', ls=':')
# mV_line = ax2.axvline(np.sqrt(2 * V0) / hbar, c='k', ls='--',
#                       label=r'$\sqrt{2mV_0}$')
# ax2.legend(prop=dict(size=12))
# ax2.set_xlabel('$k$')
# ax2.set_ylabel(r'$|\psi(k)|$')

V_x_line.set_data(S.x, S.V_x)


######################################################################
# Functions to Animate the plot
def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])

    title.set_text("")
    return (psi_x_line, V_x_line, title)


def animate(i):
    S.time_step(dt, N_steps)
    psi_x_line.set_data(S.x, abs(S.psi_x))
    V_x_line.set_data(S.x, S.V_x)

    title.set_text("t = %.2f" % S.t)
    return (psi_x_line, V_x_line, title)

# call the animator.
# blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)


# uncomment the following line to save the video in mp4 format.  This
# requires either mencoder or ffmpeg to be installed on your system
#anim.save('schrodinger_barrier.mp4', fps=15,
#          extra_args=['-vcodec', 'libx264'])

plt.show()
