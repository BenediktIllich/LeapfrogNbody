import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate



# constants in pc-M_sol-Ma units
G = 0.004491

def distance(p_1, p_2):
    """vector from p_1 towards p_2:"""
    return p_2 - p_1


def absolute(v):
    """Absolute of a 3-vector."""
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def grav_acc(m_j, d_ij, softening = 0):
    """Gives the gravitational acceleration between particles i and j."""
    return G * m_j * d_ij / ((absolute(d_ij) + softening)**3)


def acc_tensor(positions, masses, same_mass):
    """Finds the acceleration tensor given the positions and masses."""
    a = np.zeros((3, n, n))
    for i in range(n):
        if same_mass:
            for j in range(i+1, n):
                d_ij = distance(positions[:, i], positions[:, j])
                a[:, i, j] = grav_acc(masses[j], d_ij, softening) #acceleration on particle i from particle j
        else:
            for j in range(i+1, n):
                d_ij = distance(positions[:, i], positions[:, j])
                a[:, i, j] = grav_acc(masses[j], d_ij, softening) #acceleration on particle i from particle j
                a[:, j, i] = grav_acc(masses[i], -d_ij, softening) #acceleration on particle j from particle i
    return a
    
    
def cumulative_acc(positions, masses, i):
    """finds the total acceleration on particle i.(inefficient for in the loop!)"""
    a = np.zeros(3)
    for j in range(0, n):    #acceleration from particle j
        if j == i:
            a = a
        else:
            d_ij = distance(positions[:, i], positions[:, j])
            #print(d_ij)
            a = a + grav_acc(masses[j], d_ij)
    return a



# initial setup of the system (units!):

# all txt files contain (3, n)-arrays where n gives the number of particles
# positions
positions = np.loadtxt('Initial_Setup/initial_positions.txt')
# velocities
velocities = np.loadtxt('Initial_Setup/initial_velocities.txt')


# number of particles
n = 9
# number of timesteps
T = 100
# length of one time step (units!)
t = 1 #Ma
# accelerations
accelerations = np.zeros((3, n))
# masses
#masses = np.loadtxt('Initial_Setup/masses.txt')
masses = np.ones(n)
same_mass = True
# softening parameter (units!)
softening = 100 #pc

# file templates
pos_template = 'Positions/k{i}positions.txt'
vel_template = 'Velocities/k{i}velocities.txt'
acc_template = 'Accelerations/k{i}accelerations.txt'
# 0-step velocities and positions
np.savetxt(vel_template.format(i=0), velocities)
np.savetxt(pos_template.format(i=0), positions)


# 0-step accelerations:
for i in range(n):
    #print(accelerations)
    accelerations[:, i] = cumulative_acc(positions, masses, i)
np.savetxt(acc_template.format(i=0), accelerations)

# outer loop for every time step:
for k in range(1, T):
    # update the position for all particles
    positions += t * velocities + .5 * t**2 * accelerations
    # save the new positions array
    np.savetxt(pos_template.format(i=k), positions)
    
    #remember the last acceleration
    accelerations_prior = accelerations
    
    # accelerations using acceleration tensor
    a = acc_tensor(positions, masses)
    accelerations = np.sum(a, axis = 2)
    
    # save the new accelerations array
    np.savetxt(acc_template.format(i=k), accelerations)
    
    # update the velocities for all particles
    velocities += .5 * t * (accelerations_prior + accelerations)
    # save the new velocities array
    np.savetxt(vel_template.format(i=k), velocities)
