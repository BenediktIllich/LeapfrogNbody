import numpy as np
import matplotlib.pyplot as plt
import math
import gif
from IPython.display import clear_output


# Plot the Results projected into the x-y-plane
@gif.frame
def plot(i):
    positions = np.loadtxt('Positions/k{i}positions.txt'.format(i=i))
    x_positions = positions[0, :]
    y_positions = positions[1, :]
    z_positions = positions[2, :]
    plt.figure(figsize=(10,10))
    plt.scatter(x_positions, y_positions, s=8)
    plt.xlim((-2e6, 2e6))
    plt.ylim((-2e6, 2e6))
    
def radii():
    radiuses = np.zeros((n, T))
    for i in range(T):
        positions = np.loadtxt('Positions/k{i}positions.txt'.format(i=i))
        for j in range(n):
            radiuses[j, i] = absolute(positions[:, j])
    return radiuses

frames = []
T = 10000
for i in range(int(T/100)):
    frame = plot(i*100)
    frames.append(frame)
    clear_output(wait=True)
    print(f'progress: {round((i+1)*100/(T) * 100, 3)}%')
    
gif.save(frames, 'Gifs/n_body_system.gif', duration=20, unit='s', between='startend')
