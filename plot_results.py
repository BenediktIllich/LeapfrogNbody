import numpy as np
import matplotlib.pyplot as plt
import math
import gif


# Plot the Results projected into the x-y-plane
@gif.frame
def plot(i):
    positions = np.loadtxt('Positions/k{i}positions.txt'.format(i=i))
    x_positions = positions[0, :]
    y_positions = positions[1, :]
    z_positions = positions[2, :]
    plt.figure(figsize=(10,10))
    plt.scatter(x_positions, y_positions, s=8)
    plt.xlim((-5e14, 5e14))
    plt.ylim((-5e14, 5e14))
    
def radii():
    radiuses = np.zeros((n, T))
    for i in range(T):
        positions = np.loadtxt('Positions/k{i}positions.txt'.format(i=i))
        for j in range(n):
            radiuses[j, i] = absolute(positions[:, j])
    return radiuses

frames = []
for i in range(T):
    frame = plot(i)
    frames.append(frame)
    
gif.save(frames, 'Gifs/n_body_system.gif', duration=20, unit='s', between='startend')
