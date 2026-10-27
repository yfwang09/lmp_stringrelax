import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.tri import Triangulation
from   matplotlib.colors import Normalize
from regex import R
from   scipy.interpolate import interp1d
from   scipy.optimize import approx_fprime
import time

################################################################################
# Functions for the 2d potential surface

a, b, c = 0.05, 0.8, 0.05
dAB, dBC, dAC = 4.746, 4.746, 3.445
r0, alpha = 0.742, 1.942

e1, e2 = 2, 1
q1, q2 = 3/2, 1
j1, j2 = 1/2, 3
QAB = lambda r: dAB/2 * (q1*np.exp(-e1*alpha*(r-r0)) - q2*np.exp(-e2*alpha*(r-r0)))
QBC = lambda r: dBC/2 * (q1*np.exp(-e1*alpha*(r-r0)) - q2*np.exp(-e2*alpha*(r-r0)))
QAC = lambda r: dBC/2 * (q1*np.exp(-e1*alpha*(r-r0)) - q2*np.exp(-e2*alpha*(r-r0)))
JAB = lambda r: dAB/2 * (j1*np.exp(-e1*alpha*(r-r0)) - j2*np.exp(-e2*alpha*(r-r0)))
JBC = lambda r: dBC/2 * (j1*np.exp(-e1*alpha*(r-r0)) - j2*np.exp(-e2*alpha*(r-r0)))
JAC = lambda r: dAC/2 * (j1*np.exp(-e1*alpha*(r-r0)) - j2*np.exp(-e2*alpha*(r-r0)))

def VLEPS(rAB, rBC):
    rAC = rBC + rAB
    Qv = QAB(rAB)/(1+a) + QBC(rBC)/(1+b) + QAC(rAC)/(1+c)
    JABv, JBCv, JACv = JAB(rAB)/(1+a), JBC(rBC)/(1+b), JAC(rAC)/(1+c)
    Jv = JABv**2 + JBCv**2 + JACv**2 - JABv*JBCv - JBCv*JACv - JABv*JACv
    return Qv - np.sqrt(Jv)

def V(rAB, x):
    # rAC, kc, c = 3.742, 0.2025, 1.154
    rAC, kc, c = 4., 0.02025, -.154
    return VLEPS(rAB, rAC-rAB)+2*kc*(rAB-(rAC/2-x/c))**2

def G1(x, y):
    A = 1.5
    x0, y0 = 2.02083, -1.72881
    sx, sy = 0.1, 0.35
    return A * np.exp(-(x-x0)**2/2/sx) * np.exp(-(y-y0)**2/2/sy)

def G2(x, y):
    A = 6.0
    x0, y0 = 0.8, 0.1
    sx, sy = 5.0, 0.7
    return A * np.exp(-(x-x0)**2/2/sx) * np.exp(-(y-y0)**2/2/sy)

def Vnew(xy):
    x = xy[..., 0]
    y = xy[..., 1]
    return V(x, y) + G2(x, y)

def Vxy(x, y):
    return V(x, y) + G2(x, y) + 1

def Gz(x, y, z):
    A = 10
    # z0 = V(x, y)*G2(x, y)
    z0 = np.sin(x)*np.sin(y)*2
    z0 = 0
    return A*(z-z0)**2 + 1

def Vnew_3d(xyz):
    thetax = np.pi/10
    thetay = np.pi/15*0
    thetaz = np.pi/20*0
    Rx = np.array([[1, 0, 0], [0, np.cos(thetax), -np.sin(thetax)], [0, np.sin(thetax), np.cos(thetax)]])
    Ry = np.array([[np.cos(thetay), 0, np.sin(thetay)], [0, 1, 0], [-np.sin(thetay), 0, np.cos(thetay)]])
    Rz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0], [np.sin(thetaz), np.cos(thetaz), 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    xyz = xyz @ R
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    return Vxy(x, y)*Gz(x, y, z)

def get_XYZ():
    X, Y = np.meshgrid(np.linspace(0, 4, 101), np.linspace(-3, 3, 101))
    XY = np.stack([X, Y], axis=-1)
    print(XY.shape)
    Z = Vnew(XY) # V(X, Y) + G2(X, Y)
    return (X, Y, Z)

def visualize_PEL(figax=None, show=True, levels=np.linspace(-2, 10, 13)):
    X, Y, Z = get_XYZ()

    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(3.5, 4))
    ct = ax.contourf(X, Y, Z, levels=levels)
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    cbar = plt.colorbar(ct)
    cbar.set_label('$V$', fontsize=14)
    ax.axis('equal')

    if show:
        plt.show()

    return (fig, ax)

def get_XYV(z):
    xvals = np.linspace(0, 4, 101)
    yvals = np.linspace(-3, 3, 101)
    X, Y = np.meshgrid(xvals, yvals)

    XYZ = np.stack([X, Y, np.ones_like(Y)*z], axis=-1)
    V = Vnew_3d(XYZ)
    return (X, Y, V)

def get_XYZV():
    xvals = np.linspace(0, 4, 101)
    yvals = np.linspace(-3, 3, 101)
    zvals = np.linspace(-1, 1, 101)
    X, Y, Z = np.meshgrid(xvals, yvals, zvals)

    XYZ = np.stack([X, Y, Z], axis=-1)
    V = Vnew_3d(XYZ)
    return (X, Y, Z, V)

def visualize_PEL_3d(nslices=5, figax=None, show=True, levels=np.linspace(0, 14, 15), alpha=0.3):
    if figax is not None:
        fig, ax = figax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    zvals = np.linspace(-1, 1, nslices)
    vmin, vmax = levels.min(), levels.max()
    for z in zvals:
        X, Y, V = get_XYV(z)
        Z = np.ones_like(Y)*z
        zval_range = np.clip(V, vmin, vmax)
        norm = Normalize(vmin=vmin, vmax=vmax)
        surf = ax.plot_surface(X, Y, Z, linewidth=0, edgecolor="None", facecolors=plt.cm.viridis(norm(zval_range)), alpha=alpha)

    ax.axis('equal')
    if show:
        plt.show()

    return (fig, ax)

def visualize_PEL_3d_zslice(z=0, figax=None, show=True, levels=np.linspace(0, 14, 15)):
    X, Y, Z = get_XYV(z)

    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(3.5, 4))
        ct = ax.contourf(X, Y, Z, levels=levels)
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        cbar = plt.colorbar(ct)
        cbar.set_label('$V$', fontsize=14)
        ax.axis('equal')

    if show:
        plt.show()

    return (fig, ax)

################################################################################
# Functions for the stringrelax algorithm

def reparamChain(Rchain):
    ''' Reparametrize the chain
    Parameters
    ----------
    Rchain : ndarray of shape (ncpu, ndim)
        ncpu copies of configurations on the string
        ndim defines the dimension of the system
    '''
    ncpu = Rchain.shape[0]
    dR = np.diff(Rchain, axis=0)                        # (ncpu - 1, ndim)
    dRl = np.linalg.norm(dR.reshape(ncpu-1, -1), axis=-1) # (ncpu - 1, )
    Rl = np.cumsum(np.insert(dRl, 0, 0))                # (ncpu, )
    finterp = interp1d(Rl, Rchain, axis=0)
    xnew = np.arange(1, ncpu-1)/(ncpu-1)*Rl[-1]         # (ncpu - 2, )
    Rchain[1:-1, ...] = finterp(xnew)                   # (ncpu - 2, ndim)
    return Rchain

def stringrelax(rA, rB, E_func=None, F_func=None, f_epsilon=1e-5,
                nchain=6, nstep=10, relax_step=10, step_size=1e-2,
                energySpike=None, verbose=False, ):
    ''' Python implementation of stringrelax algorithm
    Parameters
    ----------
    rA, rB : ndarray of shape (ndim, )
        Initial and final configurations
    E_func, F_func : function
        Function handles calculating the energy and force
        given a configuration r of shape (ndim, )
        must provide at least one function
    f_epsilon : float
        Step size for numerical estimation of the gradient
    nchain : int
        Number of images on the string
    nstep : int
        Number of stringrelax steps (relax + reparameterization)
    relax_step : int
        Number of steps of the relaxation, currently using gradient descent
    step_size : float
        Multiplier for relaxation steps
    energySpike : float or None
        Extra relax steps if the energy path has a spike (energySpike*Emax)
        default set to None to turn off
    verbose : bool
        Print necessary information
    Return
    ----------
    Rp : ndarray of shape (nchain, ndim)
        Final result of the stringrelax algorithm
    Eclist : ndarray of shape (nstep, nchain+3)
        Energy path during the stringrelax algorithm
        Columns: (step, {energy path}, {energy of the left most node}, E0)
    '''
    if E_func is None and F_func is None:
        raise TypeError('Must provide at least one of E_func or F_func')
    if E_func is not None and F_func is None:
        F_func = lambda x: approx_fprime(x, Vnew, f_epsilon)
    
    # Initialization by linear interpolation
    Rp = np.linspace(rA, rB, nchain)
    if E_func is not None:
        E0 = E_func(rA)
        Eclist = []
    
    # Main loop for the stringrelax algorithm
    for step in range(nstep):
        if verbose:
            tic = time.time()
            print('Step %d'%step)

        # calculate initial energy in the string
        if E_func is not None:
            Ec = np.zeros(nchain)
            for i in range(nchain):
                Ec[i] = E_func(Rp[i, ...])
            EmaxDomain = Ec.argmax()
            Emax = Ec.max() - Ec[0]
            Echain_save = np.append(np.insert(Ec - Ec[0], 0, step), [Ec[0], E0])
            if verbose:
                print(Echain_save)
            Eclist.append(Echain_save)
            Ec_relaxed = Ec.copy()

        # Relax every node in the chain
        for i in range(1, nchain-1):
            for j in range(relax_step):
                grad = F_func(Rp[i, ...])
                Rp[i, ...] -= grad*step_size
            if E_func is not None:
                Ec_relaxed[i] = E_func(Rp[i, ...])

        if verbose:
            print('Relaxation: %.4fs'%(time.time()-tic))
    
        # Reparametrize the edge of the mesh
        Rp = reparamChain(Rp)

        if verbose:
            print('reparamChain: %.4fs'%(time.time()-tic))
    
    if E_func is not None:
        return Rp, np.array(Eclist)
    else:
        return Rp

################################################################################
# Functions for the meshrelax algorithm

def initTriMesh(rA, rB, rC, nside=6):
    ''' Initialize the triangular mesh with the given three points in ND space

    Parameters
    ----------
    rA, rB, rC : ndarray of shape (*ndim, )
        Three points defining the triangle
    nside : int, optional
        Number of points on each side of the triangle, by default 6

    Returns
    -------
    triObj : dict
        Dictionary containing the coordinates of the points and the uvw coordinates
        "tri" : matplotlib.tri.Triangulation
            Triangulation object
        "R" : ndarray of shape (ncpus, *ndim)
            Coordinates of the points
        "u", "v", "w" : ndarray of shape (ncpus, )
            uvw coordinates of the points
    '''

    # Initialize the uvw coordinates
    n = nside
    ncpus = int(n*(n+1)/2)                          # number of points
    xind, yind = np.tril_indices(n, k=0, m=None)
    u = np.round(1 - xind/(n-1), decimals=5)        # (ncpus, )
    v = np.round(yind/(n-1), decimals=5)            # (ncpus, )
    w = np.abs(np.round(1-u-v, decimals=5))         # (ncpus, )

    # Initialize the coordinates of the points
    ndim = rA.shape
    newshape = (ncpus, *ndim)
    rAf = rA.reshape(1, -1)                         # (1, *ndim)
    rBf = rB.reshape(1, -1)                         # (1, *ndim)
    rCf = rC.reshape(1, -1)                         # (1, *ndim)
    R_mesh = (u.reshape(-1, 1)*rAf + v.reshape(-1, 1)*rBf + w.reshape(-1, 1)*rCf).reshape(newshape)                # (ncpus, *ndim)

    tri = Triangulation(u+0.5*v, v)
    tri.x = u
    tri.y = v

    triObj = {'tri': tri, 'R': R_mesh, 'u': u, 'v': v, 'w': w}

    return triObj
