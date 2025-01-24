import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# Constants
hbar = 1.0  # Reduced Planck constant
m = 1.0     # Mass
dx = 0.1    # Grid spacing
dt = 0.01   # Time step
x_min, x_max = -10, 10  # Grid bounds
steps = 100  # Number of evolution steps

# Define the grid
x = np.arange(x_min, x_max, dx)
N = len(x)
k = fftfreq(N, d=dx) * 2 * np.pi

# Potential and initial wavefunction
def potential(x):
    return 0.5 * x**2  # Simple harmonic oscillator potential

def gaussian_wavepacket(x, x0, p0, sigma):
    return (1/(np.sqrt(sigma * np.sqrt(np.pi)))) * np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * p0 * x / hbar)

# Recursive simulation function
def evolve_wavefunction(psi, V, dt, hbar, m, recursive_depth, grid_scale=1):
    """
    Evolve a wavefunction with grid artifacts recursively layered.
    psi: Initial wavefunction.
    V: Potential energy.
    dt: Time step.
    recursive_depth: Number of recursive layers.
    grid_scale: Scaling factor for grid artifacts.
    """
    # Create artifacts through grid coarsening
    coarse_dx = dx * grid_scale
    coarse_x = np.arange(x_min, x_max, coarse_dx)
    coarse_k = fftfreq(len(coarse_x), d=coarse_dx) * 2 * np.pi

    # Recursive container for coherence tracking
    coherence_losses = []

    def single_layer_evolution(psi, x, k, V, dt):
        """Perform one evolution step using the split-operator method."""
        exp_V = np.exp(-1j * V(x) * dt / (2 * hbar))  # Half potential step
        exp_T = np.exp(-1j * (hbar**2 * k**2 / (2 * m)) * dt / hbar)  # Full kinetic step

        psi = exp_V * psi
        psi_k = fft(psi)  # Fourier transform to momentum space
        psi_k *= exp_T
        psi = ifft(psi_k)  # Inverse Fourier transform to position space
        psi *= exp_V
        return psi

    for depth in range(recursive_depth):
        # Evolve the wavefunction on the current grid
        psi = single_layer_evolution(psi, x, k, V, dt)

        # Introduce grid artifacts
        psi_coarse = np.interp(coarse_x, x, np.abs(psi)) * np.exp(1j * np.angle(np.interp(coarse_x, x, psi)))
        psi_coarse = single_layer_evolution(psi_coarse, coarse_x, coarse_k, V, dt)

        # Map coarse wavefunction back to fine grid and track coherence loss
        psi_back = np.interp(x, coarse_x, np.abs(psi_coarse)) * np.exp(1j * np.angle(np.interp(x, coarse_x, psi_coarse)))
        coherence_loss = np.sum(np.abs(psi_back - psi)) / np.sum(np.abs(psi))
        coherence_losses.append(coherence_loss)

        # Update the parent layer wavefunction
        psi = psi_back

    return psi, coherence_losses

# Initial wavefunction and potential
x0, p0, sigma = 0, 1.0, 1.0  # Initial position, momentum, and spread
psi_0 = gaussian_wavepacket(x, x0, p0, sigma)
V = potential

# Evolve the wavefunction recursively
recursive_depth = 5
psi_final, coherence_losses = evolve_wavefunction(psi_0, V, dt, hbar, m, recursive_depth, grid_scale=2)

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(psi_0)**2, label="Initial |Psi|^2")
plt.plot(x, np.abs(psi_final)**2, label="Final |Psi|^2 (Recursive)")
plt.title("Recursive Evolution of Gaussian Wavepacket")
plt.xlabel("x")
plt.ylabel("|Psi|^2")
plt.legend()
plt.grid()
plt.show()

# Coherence loss over recursive layers
plt.figure(figsize=(10, 6))
plt.plot(range(1, recursive_depth + 1), coherence_losses, marker='o')
plt.title("Coherence Loss Across Recursive Layers")
plt.xlabel("Recursive Layer")
plt.ylabel("Coherence Loss")
plt.grid()
plt.show()
