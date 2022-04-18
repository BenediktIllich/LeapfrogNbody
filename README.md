# LeapfrogNbody
gravitational n-body simulation using a leapfrog integrator

# leapfrog.py
core of the nbody algorithm. Integrates an initial setup over a number of timesteps with the possibility of having particles decay over time.
n - number of particles
T - number of Timesteps
t - physical length of one timestep
p - number of CPUs for the parallel computing of the accelerations
softening - softening length for collisionless systems
Gamma - decay crosssection for the particles (0 for stable particles)
epsilon - fraction of mass that is converted to energy during the decay
filepath - specify the path for the output

# plot_results.py
post processing of the output files. Available output flags:
plot_energies - plots the kinetic and potential enegies of a fraction of the timesteps (takes a long time)
make_position_gif - creates a movie in .gif format that shows the evolution of the system
make_density_gif - creates a gif of the mass density of the system against distance from the center for a fraction of the timesteps
plot_the_density - plots the final density and overlays it with fiducial NFW density profiles

# Generate_N_Body_Setup
samples a given geometry for initial space distribution values. converts the positions into the cartesian coordinates needed by the integrator.

# ConvertSetup
reformats other setup files so they can be used with the integrator
