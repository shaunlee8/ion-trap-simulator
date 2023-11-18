"""
Created on Fri Jul 17 07:55:50 2020

@author: Alex K.

Introduce all simulation related constants for import
"""
import numpy as np
import scipy.constants as sc
from scipy.constants import physical_constants

e = sc.e
epsilon_0 = sc.epsilon_0
k = sc.k
c = sc.c
hbar = sc.hbar
h = sc.h
R = sc.R

M = physical_constants['atomic mass constant'][0]
mb = 138*M

# Constant for calculating the coulomb force
kappa = e**2/(4*np.pi*epsilon_0)

# Hyperbolic trap parameters
r0 = 0.002
z0 = 0.0005
HVolt = 800

# RF drive frequency
Omega = 12.46e6 * 2 * np.pi

# Laser wavelength
wav = 493e-9

# Laser wave number
k_number = 2*np.pi/wav

# Laser wave vector?
dx = .1
dy = .1
dz = .04
k_vect = 1/np.sqrt(dx**2+dy**2+dz**2)*(np.array((dx,dy,dz)))

# Big K?
K = k_vect*k_number

# Excited state lifetime for p1/2 state
tau = 8.1e-9 ####I need an actual reference for this
gamma = 1/tau

# Mass of cold gas for ccd reproduction images
mg = mb / 100
T = 2e-3

# Mean and std of boltzmann distribution for virtual gas
mu = np.sqrt((8*k*T)/(np.pi * mg))
sigma = np.sqrt((k * T)/(2 * mg))
