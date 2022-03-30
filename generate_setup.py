import numpy as np
import matplotlib.pyplot as plt
import random as rd


# spherical coordinates distributions

# spherical shell distribution:
def spherical_shell(R, n):
    coordinates = np.zeros((3, n))
    for i in range(n):
        r = R + rd.uniform(-.1 * R, .1 * R)
        theta = rd.uniform(0., np.pi)
        phi = rd.uniform(0., 2 * np.pi)
        coordinates[:, i] = np.array([r, theta, phi])
    return coordinates


# solid sphere distribution
def solid_sphere(R, n):
    coordinates = np.zeros((3, n))
    for i in range(n):
        r = rd.uniform(0., R)
        theta = rd.uniform(0., np.pi)
        phi = rd.uniform(0., 2 * np.pi)
        coordinates[:, i] = np.array([r, theta, phi])
    return coordinates
    
    
# converts spherical distributions to cartesian
def spherical_to_cartesian(spherical_coordinates):
    n = np.size(spherical_coordinates, 1)
    cartesian = np.zeros((3, n))
    for i in range(n):
        x = spherical_coordinates[0, i] * np.sin(spherical_coordinates[1, i]) * np.cos(spherical_coordinates[2, i])
        y = spherical_coordinates[0, i] * np.sin(spherical_coordinates[1, i]) * np.sin(spherical_coordinates[2, i])
        z = spherical_coordinates[0, i] * np.cos(spherical_coordinates[1, i])
        cartesian[:, i] = np.array([x, y, z])
    return cartesian


def shell(R, n):
    points_on_a_sphere = spherical_to_cartesian(spherical_shell(R, n))
    x_sphere = points_on_a_sphere[0, :]
    y_sphere = points_on_a_sphere[1, :]
    z_sphere = points_on_a_sphere[2, :]

    sphere_template = 'Initial_Setup/shell_distribution_{i}_particles.txt'
    np.savetxt(sphere_template.format(i=n), points_on_a_sphere)
    
    vel_template = 'Initial_Setup/stationary_{i}_particles.txt'
    velocities = np.zeros((3, n))
    np.savetxt(vel_template.format(i=n), velocities)
    
    plt.figure(figsize=(10,10))
    plt.scatter(x_sphere, y_sphere, 1, z_sphere)
    
def sphere(R, n, M):
    points_on_a_sphere = spherical_to_cartesian(solid_sphere(R, n))
    x_sphere = points_on_a_sphere[0, :]
    y_sphere = points_on_a_sphere[1, :]
    z_sphere = points_on_a_sphere[2, :]

    sphere_template = 'Initial_Setup/sphere_distribution_{i}_particles.txt'
    np.savetxt(sphere_template.format(i=n), points_on_a_sphere)
    
    vel_template = 'Initial_Setup/stationary_{i}_particles.txt'
    velocities = np.zeros((3, n))
    np.savetxt(vel_template.format(i=n), velocities)
    
    masses = np.ones(n) * M/n
    np.savetxt(f'Initial_Setup/initial_masses_{n}_particles_M_{M}.txt', masses)
    
    #plt.figure(figsize=(10,10))
    #plt.scatter(x_sphere, y_sphere, 8)
    #plt.title('Spherical Initial Distribution', fontsize=20)
    #plt.xlabel('x/pc', fontsize=20)
    #plt.ylabel('y/pc', fontsize=20)
    #plt.xlim((-30, 30))
    #plt.ylim((-30, 30))
    #plt.savefig('sphericalinitialdistribution.png')




sphere(1e6, 100, 1e15) # Radius, n particles, total Mass
