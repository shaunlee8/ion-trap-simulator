import numpy as np
import sim_constants as const
import sim_functions as sim
import init_functions as init
from numba import njit, prange
import scipy.integrate as INT
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

# UNKNOWN IMPORT - meant for files?
# from tkinter.filedialog import askopenfilename
# import tkinter

"""
Code to draw vector arrows in the 3D plot
"""
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

def get_final_positions_and_velocities(trajectories, N):
    """
    Params
    ------
    trajectories : np.array
        a (6*N, iterations) array for the trajectories of the ions at every iteration
    N : int
        the number of ions in the trap
    
    Return
    ------
    fpos : tuple -> (np.array, np.array, np.array)
        the final positions of the ions, (xcoords, ycoords, zcoords)
    fvel : tuple -> (np.array, np.array, np.array)
        the final velocities of the ions, (xvels, yvels, zvels)
    """
    xcord = np.zeros(N, dtype=np.float64)
    ycord = np.zeros(N, dtype=np.float64)
    zcord = np.zeros(N, dtype=np.float64)
    
    xvel = np.zeros(N, dtype=np.float64)
    yvel = np.zeros(N, dtype=np.float64)
    zvel = np.zeros(N, dtype=np.float64)
    
    _, iterations = trajectories.shape
    trajs = []
    vels = []
    
    for i in range(0, N):
        # Get final position coordinates
        xcord[i] = trajectories[3*i, -1]
        ycord[i] = trajectories[3*i + 1, -1]
        zcord[i] = trajectories[3*i + 2, -1]
    
        # Get final velocity coordinates
        xvel[i] = trajectories[3*N+3*i, -1]
        yvel[i] = trajectories[3*N+3*i + 1, -1]
        zvel[i] = trajectories[3*N+3*i + 2, -1]
        
        # Get all positions
        np_trajs = np.zeros((iterations, 3))
        np_trajs[:, 0] = trajectories[3*i, :]
        np_trajs[:, 1] = trajectories[3*i + 1, :]
        np_trajs[:, 2] = trajectories[3*i + 2, :]
        trajs.append(np_trajs)

        # Get all velocities
        np_vels = np.zeros((iterations, 3))
        np_vels[:, 0] = trajectories[3*N+3*i, :]
        np_vels[:, 1] = trajectories[3*N+3*i + 1, :]
        np_vels[:, 2] = trajectories[3*N+3*i + 2, :]
        vels.append(np_vels)
    
    fpos = (xcord, ycord, zcord)
    fvel = (xvel, yvel, zvel)
    # print("final positions:", fpos)
    # print("final velocities:", fvel)
    return fpos, fvel, trajs, vels

def plot_final_ions(fpos, fvel, pscale=30e-6):
    """
    Params
    ------
    fpos : tuple -> (np.array, np.array, np.array)
        the final positions of the ions, (xcoords, ycoords, zcoords)
    fvel : tuple -> (np.array, np.array, np.array)
        the final velocities of the ions, (xvels, yvels, zvels)
    pscale : float, optional
        the scale of the 3D plot for all axes
    """
    xcord, ycord, zcord = fpos
    xvel, yvel, zvel = fvel
    ax = plt.axes(projection='3d')
    ax.set_zlabel(r'Z', fontsize=20)
    ax.set_xlim3d(-pscale, pscale)
    ax.set_ylim3d(-pscale, pscale)
    ax.set_zlim3d(-pscale, pscale)
    ax.scatter3D(xcord, ycord, zcord)

    for i in range(0, N):
        vel_i = np.zeros(3)
        vel_i[0], vel_i[1], vel_i[2] = xvel[i], yvel[i], zvel[i]
        vel_i *= pscale
        ax.arrow3D(xcord[i], ycord[i], zcord[i], vel_i[0], vel_i[1], vel_i[2], alpha=0.5)

    plt.show()

def animate_trajectories(trajs, pscale=30e-6):
    def update_lines(num, trajs, lines):
        for line, traj in zip(lines, trajs):
            # print("traj:", traj.shape)
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(traj[:num, :2].T)
            line.set_3d_properties(traj[:num, 2])
        return lines

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create lines initially without data
    lines = [ax.plot([], [], [], alpha=0.5, lw=1)[0] for _ in trajs]
    num_steps, _ = trajs[0].shape

    # Setting the axes properties
    ax.set(xlim3d=(-pscale, pscale), xlabel='X')
    ax.set(ylim3d=(-pscale, pscale), ylabel='Y')
    ax.set(zlim3d=(-pscale, pscale), zlabel='Z')

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, frames=num_steps, fargs=(trajs, lines), interval=1)

    plt.show()

def plot_individual_trajectories(trajs):
    i = 1
    for traj in trajs:
        avg_std = (np.std(traj[:, 0]) + np.std(traj[:, 1]) + np.std(traj[:, 2])) / 3
        print("average position deviation for ion {}: {}".format(i, avg_std))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Particle Trajectory")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.set_title("Particle Trajectory in 3D")
        ax.legend()
        plt.show()
        i += 1

def plot_displacement(trajs, coordinate: str, ion_num=0, pscale=10e-6, itr_slice=None, frame_metadata=None):
    coord_to_idx = {'x': 0, 'y': 1, 'z': 2}
    coord = coord_to_idx.get(coordinate, None)
    if coord is None:
        print("invalid coordinate specified, defaulting to x-displacement")
        coordinate = 'x'
        coord = 0
    iterations, _ = trajs[0].shape
    max_inum = len(trajs)
    if max_inum < ion_num:
        print("the maximum number of ions is {}".format(max_inum))
        print("defaulting to ion 1")
        ion_num = 0
    if itr_slice is not None:
        start_itr, end_itr = itr_slice[0], itr_slice[1]
        pscale = None
    else:
        start_itr, end_itr = 0, iterations

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(start_itr, end_itr), trajs[ion_num][start_itr:end_itr, coord], label="{} displacement".format(coordinate))
    if pscale is not None:
        plt.ylim(-pscale, pscale)
    plt.ylabel(coordinate + " displacement")
    plt.xlabel("time")
    plt.title("Ion {}'s {} displacement".format(ion_num + 1, coordinate))
    plt.legend()
    plt.grid()
    plt.show()

    if frame_metadata is not None:
        get_fft(trajs[ion_num][start_itr:end_itr, coord], frame_metadata, ion_num)

def get_fft(x, frame_metadata, ion_num=0):
    # doesn't seem to work, why? is the sampling rate too low?
    # x is a np.array
    n_samples, total_time = frame_metadata
    true_framerate = n_samples / total_time
    w = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1/true_framerate)
    print("xsize, nsamples", (x.size, n_samples))
    print("freqs:", freqs)
    # hz_freqs = freqs * true_framerate
    # amps = 2 / x.size * np.abs(w)
    # print("hzfreqs, amps", (hz_freqs, amps))

    plt.plot(freqs, w.real, label='real')
    plt.plot(freqs, w.imag, label='imag')
    plt.title("Real and img parts, Ion {}".format(ion_num + 1))
    plt.legend()
    plt.show()

    # plt.plot(hz_freqs[1:len(hz_freqs) // 2], amps[1:len(amps) // 2])
    plt.plot(freqs[:len(freqs) // 2], np.abs(w)[:len(w) // 2])
    # plt.plot(hz_freqs[1:1000], amps[1:1000])
    plt.title("Frequency vs. Amplitude, Ion {}".format(ion_num + 1))
    plt.show()

def get_rms_velocity(v):
    # v is an np.array
    return np.sqrt(np.mean(v ** 2))

def get_temp_barium(v_rms):
    # v_rms is a float, WORKS?
    return (v_rms ** 2) * 137.33 / (3 * const.R * 1000)

def single_shot_run():
    # TODO: messy, clean up

    # Load a file. If True you will get prompted to open one. if False, it will use random initial conditions
    load_file = False

    # Set file name for saving after
    path = 'change_name.npy'

    # Number of ions to model (7 is stable for default params)
    N = 7

    """
    Integration Parameters
    """
    # Integration step
    # t_int=5e-9

    # set integration time to 1/20 of period
    t_int = 1/(const.Omega/2/np.pi)/20

    # If not loading from a file: -------
    # total time
    Tfinal = 0.005

    # Timestep at which to record data (cant be lower than t_int and should be a multiple of it)
    t_step = 2 * t_int

    # Time variable to start at (so you don't record the whole cooling part if you don;t want to)
    t_start = 0

    #Times at which to integrate
    trange = [0, Tfinal]

    # Times to record data
    t2 = np.arange(t_start, Tfinal, t_int)
    # ------------

    """
    Initial Conditions Parameters
    """
    # Initial conditions of barium ions, temperature, boltzman mean and std
    Tb = 150

    # Size of grid where initial positions may start
    start_area = 200e-6

    # Random initial conditions
    IC = init.initialize_ions(N, start_area, Tb)

    start = time.time()
    print("simulation has started")

    # Using the leap frog algorithm (simulate trajectories with no micromotion)
    time_elapsed, trajectories = sim.leap_frog(N, Tfinal, t_int, np.array(IC))
    # print("time elapsed:", time_elapsed)
    # print("trajectories:", trajectories)

    print("simulation has finished and took", time.time() - start, "s")
    fpos, fvel, trajs, vels = get_final_positions_and_velocities(trajectories, N)

    # animate_trajectories(trajs)
    # plot_individual_trajectories(trajs)
    # plot_final_ions(fpos, fvel)

    # plot the overall displacement
    # for i in range(N):
    #     if i > 1:
    #         break
    #     plot_displacement(trajs, 'y', i, frame_metadata=None)
        # plot_displacement(trajs, 'y', i, frame_metadata=(t2.size, Tfinal))
        # plot a slice of the displacement
        # plot_displacement(trajs, 'y', i, itr_slice=[500000, 520000], frame_metadata=(t2.size, Tfinal))

    # check velocities
    temps = []
    for i in range(N):
        v_rms = get_rms_velocity(vels[i][:, 0])
        temps.append(get_temp_barium(v_rms))
    print("temps:", temps)
    return temps
    

if __name__ == '__main__':
    shots = 10
    avg_temps = []
    for i in range(shots):
        temps = single_shot_run()
        avg_temps.append(sum(temps) / len(temps))
    print("avg temps", avg_temps)
    print("avg across shots", sum(avg_temps) / len(avg_temps))
    
