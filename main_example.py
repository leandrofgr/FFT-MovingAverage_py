
import matplotlib.pyplot as plt

from FFT_MovingAverage import *


''' 
IMPORTANT: the simulations performed by the FFT_MA_3D function are periodic due to the periodic assumption of FFT.
% The usual way to overcome this fact is to generate a simulation larger
% than your model grid and then crop it.
% See Example 2 below as an example. 
'''

## EXAMPLE 1 ##

I = 100
J = 100
K = 1

noise = np.random.randn(I, J, K)

correlation_function_exp = construct_correlation_function(10, 10, noise, 1)
simulation_exp = FFT_MA_3D( correlation_function_exp, noise )

correlation_function_gau = construct_correlation_function(10, 10, noise, 2)
simulation_gau = FFT_MA_3D( correlation_function_gau, noise )

correlation_function_sph = construct_correlation_function(10, 10, noise, 3)
simulation_sph = FFT_MA_3D( correlation_function_sph, noise )


plt.figure()

# Plotting the first subplot
plt.subplot(131)
plt.imshow(simulation_exp)
plt.title('simulation_exp')

# Plotting the second subplot
plt.subplot(132)
plt.imshow(simulation_gau)
plt.title('simulation_gau')

# Plotting the third subplot
plt.subplot(133)
plt.imshow(simulation_sph)
plt.title('simulation_sph')

# Display the figure
plt.tight_layout()
plt.show()


## EXAMPLE 2 ##

mean_model = np.genfromtxt('mean_model.csv', delimiter=',')
mean_model = mean_model[:, np.newaxis]
std_model = np.genfromtxt('std_model.csv', delimiter=',')
std_model = std_model[:, np.newaxis]

# Increasing the grid model to avoid periodicity
I = int(2 * mean_model.shape[0])
J = int(1.5 * mean_model.shape[1])
K = int(1.5*mean_model.shape[2])

noise = np.random.randn(I, J, K)
correlation_function_sph = construct_correlation_function(10, 10, noise, 3)
simulation_sph = FFT_MA_3D( correlation_function_sph, noise )

# croping the simulation to avoid periodicity
simulation_sph = simulation_sph[0:int(I/2),0:1,0:int(K/1.5)]

elevation_simulation = mean_model + simulation_sph * std_model

plt.figure()
plt.imshow(np.squeeze(elevation_simulation, axis=1))
plt.tight_layout()
plt.show()