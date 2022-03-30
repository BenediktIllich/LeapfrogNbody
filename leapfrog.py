import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate
import random
from IPython.display import clear_output
import time
from numba import jit


start = time.time()

# constants in pc-M_sol-Ma units
G = 0.004491
c = 306000 # pc/Ma

def distance(p_1, p_2):
    """vector from p_1 towards p_2:"""
    return p_2 - p_1


def absolute(v):
    """Absolute of a 3-vector."""
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def grav_acc(m_j, d_ij, softening = 0):
    """Gives the gravitational acceleration between particles i and j."""
    return G * m_j * d_ij / ((absolute(d_ij) + softening)**3)


def acc_tensor(positions, masses):
    """Finds the acceleration tensor given the positions and masses."""
    a = np.zeros((3, n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                a[:, i, j] = 0
            else:
                d_ij = distance(positions[:, i], positions[:, j])
                a[:, i, j] = grav_acc(masses[j], d_ij, softening) #acceleration on particle i from particle j
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
    
    
def get_epsilon(kick_velocity):
    """returns the parameter epsilon from a given kick velocity"""
    return kick_velocity / (kick_velocity + c)
    
    
def new_mass(epsilon):
    """returns the ratio of m_daughter to m_mother"""
    return np.sqrt(1 - 2 * epsilon)
    
    
def random_direction():
    theta = np.random.uniform(low = 0.0, high = np.pi)
    phi = np.random.uniform(low = 0.0, high = 2 * np.pi)
    return theta, phi
    

def relative_cart_kick(kick_velocity):
    theta, phi = random_direction()
    x_velocity = kick_velocity * np.cos(phi) * np.sin(theta)
    y_velocity = kick_velocity * np.sin(phi) * np.sin(theta)
    z_velocity = kick_velocity * np.cos(theta)
    return x_velocity, y_velocity, z_velocity
    
    
def decay(decay_rate, kick_velocity, mass_array, velocity_array, time_step):
    """changes the mass and velocity array according to random decay"""
    epsilon = get_epsilon(kick_velocity)
    daughter_mass_ratio = new_mass(epsilon)
    P = time_step * decay_rate # probability for a particle to decay during dt
    # generate a random number between 0 and 1 for each particle
    R = np.random.uniform(low=0.0, high=1.0, size=(len(mass_array)))
    for particle, decay_prob in enumerate(R):
        if decay_prob < P: # in this case decay happens
            #print('boom')
            mass_array[particle] = daughter_mass_ratio * mass_array[particle]
            x_kick, y_kick, z_kick = relative_cart_kick(kick_velocity)
            velocity_array[:, particle] = velocity_array[:, particle] + np.array([x_kick, y_kick, z_kick])
    return mass_array, velocity_array
            




# number of particles
n = 100
# number of timesteps
T = 1000
# length of one time step (units!)
t = 1 #Ma
# accelerations
accelerations = np.zeros((3, n))
# masses
M = 1e15
masses = np.loadtxt(f'Initial_Setup/initial_masses_{n}_particles_M_{M}.txt')
#masses = np.ones(n)
# softening parameter (units!)
softening = 1e4 #pc

# decay parameters
Gamma = 0 #Ma
v_kick = 100 #pc/Ma

# initial setup of the system (units!):
# all txt files contain (3, n)-arrays where n gives the number of particles
# positions
positions = np.loadtxt(f'Initial_Setup/sphere_distribution_{n}_particles.txt')
# velocities
velocities = np.loadtxt(f'Initial_Setup/stationary_{n}_particles.txt')

# file templates
pos_template = 'Positions/k{i}positions.txt'
vel_template = 'Velocities/k{i}velocities.txt'
acc_template = 'Accelerations/k{i}accelerations.txt'
# 0-step velocities and positions
np.savetxt(vel_template.format(i=0), velocities)
np.savetxt(pos_template.format(i=0), positions)


# 0-step accelerations:
#for i in range(n):
#    #print(accelerations)
#    accelerations[:, i] = cumulative_acc(positions, masses, i)
#np.savetxt(acc_template.format(i=0), accelerations)

a = acc_tensor(positions, masses)
accelerations = np.sum(a, axis = 2)

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

    # decay process
    masses, velocities = decay(Gamma, v_kick, masses, velocities, t)
    
    # progress update
    clear_output(wait=True)
    print(f'progress: {round((k+1)/(T) * 100, 3)}%')

end = time.time()

print(f'Simulation finished. It took {end - start} seconds')
