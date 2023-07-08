import numpy as np
import sim_constants as const
from numba import njit, prange

@njit
def Coulomb_vect(X, N):
    """
    Takes a vector of ion positions in Nx1 format and calculates the coulomb force on each ion.
    
    Params
    ------
    X : np.array
        a vector of ion positions
    N : int
        the number of ions in the trap
    
    Return
    ------
    F : np.array
        a 3xN array of coulomb force vectors for N ions
    """
    F = np.zeros((N,3))
    X3 = X.reshape(N,3)
     
    for i in prange(1, N):
        # Calculate the permutation
        perm = X3 - np.roll(X3, 3*i)

        # Calculate the coulomb forces
        F += const.kappa*perm/((np.sqrt((perm**2).sum(axis=1)).repeat(3).reshape((N,3))))**3
        
    # Return flattened array
    return F.ravel()

@njit
def F_DC(X, N):
    """
    Creates a fake dc force (to be replaced by real one soon) 
    somehow setting values too high can cause simulation not to work
    
    Params
    ------
    X : np.array
        a vector of ion positions
    N : int
        the number of ions in the trap
    
    Return
    ------
    F : np.array
        a 3xN array of DC force vectors for N ions
    """
    # Reshape input array
    X3 = X.reshape((N,3))
    
    # Initialiize the output array
    F = np.zeros((N,3))
    
    # Choose amount to increase/decrease forces by
    wx = -.14e6*2*np.pi
    wy = .14e6*2*np.pi
    wz = .8e6*np.pi
    
    # Set the order of forces
    trap = np.array([wx,wy,wz])
    
    # Set the sign of forces
    order = np.array([1,-1,1])
    
    # Calculate the force
    F -= const.mb*trap**2*order*X3
    
    # Return the flattened array
    return F.ravel()

@njit
def FHYP_pseudo(X,N):
    """
    which force is this?
    
    Params
    ------
    X : np.array
        a 3xN vector of ion positions
    N : int
        the number of ions in the trap
    Return
    ------
    F : np.array
        a 3xN array of force vectors for N ions
    """
    # Reshape the coordinate array to xyz coords of each ion
    Y = X.reshape((N,3))
    
    # Separate the coordinates
    x = Y[:,0]
    y = Y[:,1]
    z = Y[:,2]
    coeff = const.e**2*const.HVolt**2/(const.mb*const.Omega**2*(const.r0**2+2*const.z0**2)**2)
    
    # Convert to polar coordinates
    R = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
      
    FR = coeff*-2*R
    FZ = -8*coeff*z
      
    F = np.stack((FR*np.cos(phi),FR*np.sin(phi),FZ))
    
    return F.transpose(1,0).ravel()

@njit
def laser_vect(V,N):
    """
    uses foote et al paper for reference, what is this?
    
    Params
    ------
    V : ???
    N : ???
    
    Return
    ------
    F : np.array
        a 3xN array of force vectors for N ions
    """
    # Initialize output array
    F = np.zeros((N,3))
    
    # What are these?
    vel = V.reshape((N,3))
    s0 = 0.25
    F0 = 0
    
    # Project velocity into laser direction
    speedk = -vel.dot(const.k_vect)
    delta = -200e6*2*np.pi+const.k_number*speedk*0
    delta = 0
    S = s0/(1+(2*delta/const.gamma)**2)
    S = np.zeros((N,1))
    
    # leave out F0 for now HUH???? its right there????
    F0 = const.hbar*const.K*const.gamma*S/2/(1+S)
    F += F0
    
    # Calculate recoil
    F += const.hbar*S/(1+S)*(.5*const.gamma*const.K)
      
    # Flatten array
    F += np.kron(S,const.k_vect) # what is np.kron?????
    F = F.ravel()
    
    # Damping coefficient
    F.ravel() # inplace double ravel ???????
    Beta = -const.hbar*4*s0*delta/const.gamma/(1+s0+(2*delta/const.gamma)**2)**2*np.kron(vel.dot(const.K),const.K)
    F -= Beta*np.kron(vel.dot(const.k_vect),const.k_vect)
    
    return F