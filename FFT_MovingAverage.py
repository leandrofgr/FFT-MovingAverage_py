import numpy as np
from scipy.fftpack import fftn, ifftn


def construct_correlation_function(Lv, Lh, signal, type):

    I = signal.shape[0]
    J = signal.shape[1]
    K = signal.shape[2]

    # Taper parameters (it avoids artifacts when the variogram range is similar to the model size):
    order = 4
    desvio = 1.0

    correlation_function = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if type == 1:
                    value = np.exp(-np.sqrt((((i - np.round(I/2))**2) / Lv**2) + (((j - np.round(J/2))**2) / Lh**2) + (((k - np.round(K/2))**2) / Lh**2)))
                elif type == 2:
                    value = np.exp(-((((i - np.round(I/2))**2) / Lv**2) + (((j - np.round(J/2))**2) / Lh**2) + (((k - np.round(K/2))**2) / Lh**2)))
                elif type == 3:
                    r = np.sqrt((((i - np.round(I/2)) / (3 * Lv))**2) + (((j - np.round(J/2)) / (3 * Lh))**2) + (((k - np.round(K/2)) / (3 * 2 * Lh))**2))
                    if r < 1:
                        value = 1 - 1.5 * r + 0.5 * r**3
                    else:
                        value = 0

                value_window = np.exp(-((np.abs((i - np.round(I/2))) / (desvio * I))**order + (np.abs((j - np.round(J/2))) / (desvio * J))**order + (np.abs((k - np.round(K/2))) / (desvio * K))**order))
                correlation_function[i, j, k] = value * value_window
                # correlation_function[i, j, k] = value

    return correlation_function
	
	


def FFT_MA_3D(correlation_function, noise):
    # correlation_function: Correlation/variogram model, output of the function "construct_correlation_function".
    # noise: White random noise that will be filtered.

    filter = correlation_function
    simulation = np.real(ifftn(np.sqrt(np.abs(fftn(filter, noise.shape))) * fftn(noise, noise.shape)))

    return simulation	