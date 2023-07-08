import numpy as np
import sim_constants as const
import init_functions as init
import force_functions as force
from numba import njit

@njit
def collisions(V):
    """
    UNKNOWN?
    
    Params
    ------
    V : ???
    
    Return
    ------
    X : ???
    """
    X = np.zeros(len(V))
    N = int(len(V)/3)
    for i in range(0,N):
        Y = V[3*i:3*i+3]
        X[3*i:3*i+3] = collision(Y)
    return X

@njit
def collision(v_old):
    """
    UNKNOWN?
    
    Params
    ------
    v_old : ???
    
    Return
    ------
    ??? : ???
    """
    vg = init.rand_v(const.mu,const.sigma)
    n0 = init.rand_3D_vect() ####random velocity unit vector assigned to ion
    n1 = init.rand_3D_vect() ####random velocity of gas particle
    return const.mg/(const.mb+const.mg)*np.abs(np.linalg.norm(v_old)-vg)*n0+(const.mb*v_old+const.mg*vg*n1)/(const.mb+const.mg)

@njit
def leap_frog(N, T, tstep, IC):
    """
    UNKNOWN?
    
    Params
    ------
    N : int
        the number of ions to simulate in the trap
    T : float
        the temperature of the trap/ions?
    tstep : float
        the integration time of the sensor
    IC : np.array
        a 6xN array of concatenated position and velocity vectors
    
    Return
    ------
    time_elapsed : np.array
        redundant?
    trajectories : np.array
        a (6, N, iterations) array for the trajectories of the ions at every iteration 
    """
    # Mass of barium?
    mb = const.mb
    
    # Determine the number of iterations
    iterations = int(T / tstep)
    time_elapsed = np.zeros(iterations)
    
    # Initialize trajectories
    trajectories = np.zeros((6*N,iterations), dtype=np.float64)
    trajectories[:,0] = IC
    pos = np.zeros(3*N)
    vel = np.zeros(3*N)

    # Initialize acceleration variable. keep track of old/new
    acc = np.zeros((3*N,2))
    t = 0
    pos = trajectories[0:3*N,0].copy()
    vel = trajectories[3*N:6*N,0]
    acc[:,1] = force.Coulomb_vect(pos, N)/const.mb + force.F_DC(pos,N)/const.mb + force.FHYP_pseudo(pos, N)/const.mb
    it = 1
    t = tstep
      
    while it < iterations:
        # Get current position, velocity of all ions
        acc[:,0] = acc[:,1].copy()
        pos = trajectories[0:3*N,it-1].copy()
        vel = trajectories[3*N:6*N,it-1].copy()
        
        # Update positions based on x=vit+1/2at^2
        trajectories[0:3*N,it] = pos+vel*tstep+.5*tstep**2*acc[:,0].copy()
        ########record old acceleration, calculate new one
          
        # Sum up all forces:
        # with micromotion
        acc[:,1] = 1/const.mb*(force.Coulomb_vect(pos,N) + force.F_DC(pos,N)+force.FHYP_pseudo(pos,N))+force.laser_vect(vel,N)
        
        # without micromotion
        # acc[:,1]=1/mb*(Coulomb_vect(pos,N)  +F_DC(pos,N)+FHYP_pseudo(pos,N) )

        # Compute velocities
        trajectories[3*N:6*N,it] = vel+.5*tstep*(acc[:,0].copy()+acc[:,1].copy())
        trajectories[3*N:6*N,it] = collisions(trajectories[3*N:6*N,it].copy())
        time_elapsed[it] = t
        it += 1
        t += tstep
        
    return time_elapsed, trajectories