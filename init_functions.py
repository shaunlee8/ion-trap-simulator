import numpy as np
import sim_constants as const
from numba import njit

@njit
def rand_v(mu, sigma):
    """
    A random speed generator based on light gas particles at a desired temperature with mean and std 
    given by the maxwell boltzman distribution.
    
    Params
    ------
    mu : float
        the mean of the maxwell-boltzman distribution
    sigma : float
        the standard deviation of the maxwell-boltzman distribution
    
    Return: float
        a scalar value for the speed drawn from a normal distribution
    """
    # The distribution in terms of the vectorial velocity is a normal distribution
    return np.random.normal(mu, sigma)
    
@njit
def rand_3D_vect():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution.
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution.
    
    Return
    ------
    vect : np.array
        a 3x1 vector array with magnitude 1 
    """
    vect = np.zeros(3)
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    
    vect[0] = np.sin(theta) * np.cos(phi)
    vect[1] = np.sin(theta) * np.sin(phi)
    vect[2] = np.cos(theta)
    return vect

def rand_pos(grid_size):
    """
    A random position generator.
    
    Params
    ------
    grid_size : float
        the scale of the grid of coordinates for the trap
    
    Return: np.array
        a 3x1 array filled with random positions chosen from the interval [0, 0.5 * grid_size)
    """
    return (np.random.random_sample(3) - 0.5) * grid_size

def initialize_ions(N, start_area, T):
    """
    Initializes 
    
    Params
    ------
    N : int
        the number of barium ions to initialize
    start_area : float
        the grid size of the trap
    T : float
        the temperature of the barium ions
    
    Return
    ------
    IC : np.array
        a 6xN array containing the concatenated position and velocity vectors for each ion.
        positions are indexed at IC[3*i:3*i+2] and velocities are indexed at IC[3*N+3*i:3*N+3*i+2].
        i.e. for ions 1 and 2, IC = [x_1, y_1, z_1, x_2, y_2, z_2, vx_1, vy_1, vz_1, vx_2, vy_2, vz_2]
    """
    mu_barium = np.sqrt((8*const.k*T)/(np.pi*const.mb))
    sigma_barium = np.sqrt((const.k*T)/(2*const.mb))
    # Initialize the output array
    IC = np.zeros(6*N)
    
    # Loop through the ions
    for i in range(1,N+1):
        # TODO: does this really provide the same computational efficiency as C?
        # Init random position for ion i
        IC[3*(i-1):3*(i-1)+3] = rand_pos(start_area)
        
        # Init random velocity for ion i
        IC[3*N+3*(i-1):3*N+3*(i-1)+3] = rand_3D_vect() * rand_v(mu_barium, sigma_barium)
        
    return IC