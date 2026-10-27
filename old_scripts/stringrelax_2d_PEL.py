import numpy as np
from   scipy.optimize import minimize, approx_fprime
import matplotlib.pyplot as plt

from   stringrelax_util import Vnew, visualize_PEL, stringrelax

# Get the three minima on the 2d potential surface
xy0_list = [[0.5, 0], [3, -1.2], [0.5, 4], ]
labels = ['A', 'B', 'C']

x_min = []
for i in range(3):
    xy0 = xy0_list[i]
    label = labels[i]
    res = minimize(Vnew, xy0, method='CG')
    x_min.append(res.x)
    print('%s: V(%8.4f,%8.4f) = %.8f'%(label, res.x[0], res.x[1], res.fun))
x_min = np.array(x_min)

# Visualize the three minima
fig, ax = visualize_PEL(show=False)
ax.plot(x_min[:, 0], x_min[:, 1], 'ow')
ax.text(x=x_min[0,0], y=x_min[0,1]-0.6, s='A', fontsize=16, color='w')
ax.text(x=x_min[1,0], y=x_min[1,1]-0.6, s='B', fontsize=16, color='w')
ax.text(x=x_min[2,0], y=x_min[2,1]+0.3, s='C', fontsize=16, color='w')
plt.show()

################################################################################
# Example 1. Calculate and visualize the MEP using only the energy function
nchain = 40
Rp, Eclist = stringrelax(x_min[0, ...], x_min[1, ...], E_func=Vnew, nchain=nchain)

# Visualize the path spatially
fig, ax = visualize_PEL(show=False)
ax.plot(x_min[:, 0], x_min[:, 1], 'ow')
ax.plot(Rp[:, 0], Rp[:, 1], '-w')
ax.text(x=x_min[0,0], y=x_min[0,1]-0.6, s='A', fontsize=16, color='w')
ax.text(x=x_min[1,0], y=x_min[1,1]-0.6, s='B', fontsize=16, color='w')
ax.text(x=x_min[2,0], y=x_min[2,1]+0.3, s='C', fontsize=16, color='w')
fig.tight_layout()

# Visualize the final energy path (reference energy at state A)
print(Eclist.shape)
EA = Eclist[1, -1] # energy of state A
figE, axE = plt.subplots()
axE.plot(Eclist[-1, 1:-2])
axE.set_xlabel('Reaction coordinate')
axE.set_ylabel(r'V ($E_A=%.4f$)'%EA)

plt.show()

################################################################################
# Example 2. Calculate and visualize the MEP using only the energy function

# Numerical force handle
epsilon = 1e-5
force = lambda x: approx_fprime(x, Vnew, epsilon)

nchain = 40
Rp = stringrelax(x_min[0, ...], x_min[1, ...], F_func=force, nchain=nchain)

fig, ax = visualize_PEL(show=False)
ax.plot(x_min[:, 0], x_min[:, 1], 'ow')
ax.plot(Rp[:, 0], Rp[:, 1], '-w')
ax.text(x=x_min[0,0], y=x_min[0,1]-0.6, s='A', fontsize=16, color='w')
ax.text(x=x_min[1,0], y=x_min[1,1]-0.6, s='B', fontsize=16, color='w')
ax.text(x=x_min[2,0], y=x_min[2,1]+0.3, s='C', fontsize=16, color='w')
fig.tight_layout()

figE, axE = plt.subplots()
Elast = Vnew(Rp)
axE.plot(Elast - Elast[0])
axE.set_xlabel('Reaction coordinate')
axE.set_ylabel(r'V ($E_A=%.4f$)'%EA)

plt.show()
