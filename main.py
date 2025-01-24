import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
x_min, x_max = -10, 10  # Grid range
N = 256  # Number of grid points
timesteps = 100  # Number of timesteps
dt = 0.01  # Time step size
recursion_depth = 5  # Number of recursive layers
feedback_strength = 0.5  # Strength of the feedback potential

# Define the spatial grid
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

# Initial Gaussian wavepacket
def gaussian_wavepacket(x, x0, p0, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) ** 0.5 * np.exp(-((x - x0) ** 2) / (4 * sigma ** 2) + 1j * p0 * x)

# Time evolution operators
T = np.exp(-0.5j * k**2 * dt)  # Kinetic energy operator

# Recursive evolution with feedback mechanism
def recursive_evolution(psi, depth, feedback_strength):
    if depth == 0:
        return psi

    # Density of the wavepacket
    density = np.abs(psi) ** 2

    # Define a feedback potential based on the density
    feedback_potential = feedback_strength * (density - np.mean(density))

    # Potential energy operator with feedback
    V = np.exp(-1j * feedback_potential * dt)

    # Perform split-operator step
    psi = np.fft.ifft(T * np.fft.fft(V * psi))

    # Recursive step
    psi = recursive_evolution(psi, depth - 1, feedback_strength)
    return psi

# Initial wavepacket parameters
x0, p0, sigma = 0, 1, 1
psi_initial = gaussian_wavepacket(x, x0, p0, sigma)

# Evolve the wavepacket with recursion
psi_final = recursive_evolution(psi_initial, recursion_depth, feedback_strength)

# Compute densities
initial_density = np.abs(psi_initial) ** 2
final_density = np.abs(psi_final) ** 2

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, initial_density, label="Initial |Psi|^2")
plt.plot(x, final_density, label="Final |Psi|^2 (Recursive with Feedback)")
plt.title("Recursive Evolution of Gaussian Wavepacket with Feedback")
plt.xlabel("x")
plt.ylabel("|Psi|^2")
plt.legend()
plt.grid()
plt.show()

# Coherence loss analysis
coherence_loss = []
for depth in range(1, recursion_depth + 1):
    psi_temp = recursive_evolution(psi_initial, depth, feedback_strength)
    coherence = np.sum(np.abs(np.fft.fft(psi_temp)) ** 2)
    coherence_loss.append(coherence)

plt.figure(figsize=(10, 6))
plt.plot(range(1, recursion_depth + 1), coherence_loss, marker='o')
plt.title("Coherence Loss Across Recursive Layers with Feedback")
plt.xlabel("Recursive Depth")
plt.ylabel("Coherence")
plt.grid()
plt.show()
