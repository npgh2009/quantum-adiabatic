"""
Credit: mholtrop
Edited by: npgh2009
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

#initialize harmonic oscillator parameters
hbar=1
m=1
omega=1

#initialize x grid
N = 200 # grid points
a = 20.0 # x limits
x = np.linspace(-a/2.,a/2.,N)
dx = x[1]-x[0] # step

#Potential
V = .5*m*omega*x*x
# V[N/2]=2/dx   # This would add a "delta" spike in the center.

# Construct hamiltonian matrix
DerivativeTerm = 1/(dx*dx)*sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
Hamiltonian = -(hbar*hbar)/(2.0*m)*DerivativeTerm + sp.diags(V) 

En,psiT = eigs(Hamiltonian) # This computes the eigen values and eigenvectors
psi = np.transpose(psiT) 
# The psi now contain the wave functions ordered so that psi[n] if the n-th eigen state.

#Check the normalization of the wave function arrays.
notok=False
for n in range(len(psi)):
    # s = np.sum(psi[n]*psi[n])
    s = np.linalg.norm(psi[n])  # This does the same as the line above.
    if np.abs(s - 1) > 0.00001: # Check if it is different from one.
        print("Wave function {} is not normalized to 1 but {}".format(n,s))
        notok=True

if not notok:
    print("All the psi_n(x) are normalized.")

fig2 = plt.figure(figsize=[10,7])
plt.title('Harmonic Oscillator')
plt.ylabel('$\psi(x)$')
plt.xlabel('$x$')
#plt.plot([0,0],[-6,V[0]],color="blue")
#plt.plot([-a/2.,a/2.],[0,0],color="blue")
plt.plot(x, 0.1*V, color="grey", label="V(x) scaled by 0.1")

plt.ylim(-1,1)
plt.xlim(-6,6)

for i in range(5):
    if psi[i][int(N/8)] < 0:
        plt.plot(x,-abs(psi[i])/np.sqrt(dx),label="$E_{}$={:3.1f}".format(i,En[i]))
    else:
        plt.plot(x,abs(psi[i])/np.sqrt(dx),label="$E_{}$={:3.1f}".format(i,En[i]))

plt.title("Solution to harmonic oscillator")
plt.legend()
plt.show()