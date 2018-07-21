"""
Solve and animate the Schrodinger equation

First presented at http://jakevdp.github.com/blog/2012/09/05/quantum-python/

Author: Jake Vanderplas <vanderplas@astro.washington.edu>
License: BSD

Edited by: npgh2009
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from schrodinger import Schrodinger
from HarmonicOscillator import HarmonicOscillator

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
alpha = m * omega / hbar
L = 1/np.sqrt(alpha) #length scale

# # specify initial velocity for coherence state
v0 = 0.3*omega*L #should not be much larger than omega*L to keep wavefunction shape coherent
p0 = m * v0
k0 = p0 / hbar

# specify initial wavefunction

psi_x0 = qho.eigenFunction(n = 0)
#psi_x0 = qho.coherenceState(v0, n = 0)
#psi_x0 = qho.squeezeState(omegasq = 0.1)

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
ymin = 0
ymax = 10*hbar*omega

# choose plotting functions
PROB_AMPLITUDE = 1 #probability amplitude (absolute value)
REAL_AND_IMAG = 2 #real and imaginary part
REAL_IMAG_AND_ABS = 3

PLOTFUNCTION = REAL_IMAG_AND_ABS

if PLOTFUNCTION == PROB_AMPLITUDE:
    ax1 = fig.add_subplot(111, xlim=xlim,
                                ylim=(ymin - 0.2 * (ymax - ymin),
                                ymax + 0.2 * (ymax - ymin)))
    psi_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
    V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')

    ax1.legend(prop=dict(size=12))
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$|\psi(x)|$')
    V_x_line.set_data(S.x, S.V_x)

    # Functions to Animate the plot
    def init():
        psi_x_line.set_data([], [])
        V_x_line.set_data([], [])

        return (psi_x_line, V_x_line)


    def animate(i):
        S.time_step(dt, N_steps)
        psi_x_line.set_data(S.x, 2*abs(S.psi_x))
        V_x_line.set_data(S.x, S.V_x)

        return (psi_x_line, V_x_line)

    # call the animator.
    # blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=30, blit=True)
    plt.show()

elif PLOTFUNCTION == REAL_AND_IMAG:
    ax1 = fig.add_subplot(111, xlim=xlim,
                                ylim=(ymin - 0.8 * (ymax - ymin),
                                ymax + 0.2 * (ymax - ymin)))
    psi_x_real_line, = ax1.plot([], [], c='r', label=r'$Re(\psi(x))$')
    psi_x_imag_line, = ax1.plot([], [], c='b', label=r'$Im(\psi(x))$')
    V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')

    ax1.legend(prop=dict(size=12))
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\psi(x)$')
    V_x_line.set_data(S.x, S.V_x)

    # Functions to Animate the plot
    def init():
        psi_x_real_line.set_data([], [])
        psi_x_imag_line.set_data([], [])
        V_x_line.set_data([], [])

        return (psi_x_real_line, psi_x_imag_line, V_x_line)


    def animate(i):
        S.time_step(dt, N_steps)
        psi_x_real_line.set_data(S.x, 2*np.real(S.psi_x))
        psi_x_imag_line.set_data(S.x, 2*np.imag(S.psi_x))
        V_x_line.set_data(S.x, S.V_x)

        return (psi_x_real_line, psi_x_imag_line, V_x_line)

    # call the animator.
    # blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=30, blit=True)
    plt.show()

elif PLOTFUNCTION == REAL_IMAG_AND_ABS:
    ax1 = fig.add_subplot(111, xlim=xlim,
                                ylim=(ymin - 0.8 * (ymax - ymin),
                                ymax + 0.2 * (ymax - ymin)))
    psi_x_real_line, = ax1.plot([], [], c='r', label=r'$Re(\psi(x))$')
    psi_x_imag_line, = ax1.plot([], [], c='b', label=r'$Im(\psi(x))$')
    psi_x_line, = ax1.plot([], [], c='g', label=r'$|\psi(x)|$')
    V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')

    ax1.legend(prop=dict(size=12))
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\psi(x)$')
    V_x_line.set_data(S.x, S.V_x)

    # Functions to Animate the plot
    def init():
        psi_x_real_line.set_data([], [])
        psi_x_imag_line.set_data([], [])
        psi_x_line.set_data([], [])
        V_x_line.set_data([], [])

        return (psi_x_real_line, psi_x_imag_line, psi_x_line, V_x_line)


    def animate(i):
        S.time_step(dt, N_steps)
        psi_x_real_line.set_data(S.x, 2*np.real(S.psi_x))
        psi_x_imag_line.set_data(S.x, 2*np.imag(S.psi_x))
        psi_x_line.set_data(S.x, 2*abs(S.psi_x))
        V_x_line.set_data(S.x, S.V_x)

        return (psi_x_real_line, psi_x_imag_line, psi_x_line, V_x_line)

    # call the animator.
    # blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=30, blit=True)
    plt.show()

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

######################################################################
# Functions to Animate the plot
# def init():
#     psi_x_line.set_data([], [])
#     V_x_line.set_data([], [])

#     title.set_text("")
#     return (psi_x_line, V_x_line, title)


# def animate(i):
#     S.time_step(dt, N_steps)
#     psi_x_line.set_data(S.x, abs(S.psi_x))
#     V_x_line.set_data(S.x, S.V_x)

#     title.set_text("t = %.2f" % S.t)
#     return (psi_x_line, V_x_line, title)

# # call the animator.
# # blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=frames, interval=30, blit=True)


# # uncomment the following line to save the video in mp4 format.  This
# # requires either mencoder or ffmpeg to be installed on your system
# #anim.save('schrodinger_barrier.mp4', fps=15,
# #          extra_args=['-vcodec', 'libx264'])

# plt.show()
