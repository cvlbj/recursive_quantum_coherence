import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

# Constants for the simulation
dx = 0.1  # Spatial resolution
L = 20.0  # Length of the grid
x = np.arange(-L / 2, L / 2, dx)  # Spatial grid
N = len(x)  # Number of grid points
dt = 0.01  # Time step for evolution
T_total = 1.0  # Total time of evolution
steps = int(T_total / dt)  # Number of time steps

# Recursive parameters
recursive_depth = 5  # Number of recursive layers
fine_dx = dx / 2  # Fine grid resolution for inner layers
fine_x = np.arange(-L / 2, L / 2, fine_dx)

# Define the Gaussian wavepacket
def gaussian_wavepacket(x, x0=0.0, p0=1.0, sigma=1.0):
    return np.exp(-0.5 * ((x - x0) / sigma)**2) * np.exp(1j * p0 * x)

# Potential function (free particle for simplicity)
def V(x):
    return np.zeros_like(x)

# Split-operator method for wavefunction evolution
def split_operator_step(psi, x, dx, dt):
    k = fftfreq(len(x), d=dx) * 2 * np.pi
    # Kinetic energy operator
    kinetic_phase = np.exp(-1j * (k**2) * dt / 2)
    # Potential energy operator
    potential_phase = np.exp(-1j * V(x) * dt)
    # Apply split operator method
    psi = ifft(kinetic_phase * fft(psi))
    psi *= potential_phase
    psi = ifft(kinetic_phase * fft(psi))
    return psi

# Recursive evolution function
def recursive_evolution(psi, x, dx, dt, depth, fine_x, fine_dx):
    coherence_losses = []
    intermediate_states = []

    for d in range(depth):
        # Evolve on the coarse grid
        for _ in range(steps):
            psi = split_operator_step(psi, x, dx, dt)

        # Evolve on the fine grid (inner layer)
        fine_psi = gaussian_wavepacket(fine_x)
        for _ in range(steps):
            fine_psi = split_operator_step(fine_psi, fine_x, fine_dx, dt)

        # Calculate coherence loss (overlap integral)
        overlap = np.abs(np.sum(np.conj(psi) * np.interp(x, fine_x, fine_psi)) * dx)**2
        coherence_losses.append(1 - overlap)

        # Store intermediate state
        intermediate_states.append(np.abs(psi)**2)

    return psi, coherence_losses, intermediate_states

# Introduce stochastic collapse mechanism
def stochastic_collapse(psi, x):
    prob = np.abs(psi)**2
    prob /= prob.sum()  # Normalize probabilities
    collapsed_index = np.random.choice(len(x), p=prob)
    collapsed_state = np.zeros_like(psi)
    collapsed_state[collapsed_index] = 1.0
    return collapsed_state

# Initial wavepacket
psi = gaussian_wavepacket(x)

# Perform recursive evolution
final_psi, coherence_losses, intermediate_states = recursive_evolution(
    psi, x, dx, dt, recursive_depth, fine_x, fine_dx
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(psi)**2, label="Initial |Psi|^2")
plt.plot(x, np.abs(final_psi)**2, label=f"Final |Psi|^2 (Recursive Depth {recursive_depth})")
plt.xlabel("x")
plt.ylabel("|Psi|^2")
plt.title("Recursive Evolution of Gaussian Wavepacket")
plt.legend()
plt.grid()
plt.show()

# Plot coherence loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, recursive_depth + 1), coherence_losses, marker="o")
plt.xlabel("Recursive Depth")
plt.ylabel("Coherence Loss")
plt.title("Coherence Loss Across Recursive Layers")
plt.grid()
plt.show()

# Visualize intermediate states
plt.figure(figsize=(10, 6))
for i, state in enumerate(intermediate_states):
    plt.plot(x, state, label=f"Layer {i+1}")
plt.xlabel("x")
plt.ylabel("|Psi|^2")
plt.title("Intermediate States Across Layers")
plt.legend()
plt.grid()
plt.show()

# Simulate stochastic collapse
collapsed_psi = stochastic_collapse(final_psi, x)
plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(final_psi)**2, label="Final |Psi|^2")
plt.plot(x, np.abs(collapsed_psi)**2, label="Collapsed State", linestyle="--")
plt.xlabel("x")
plt.ylabel("|Psi|^2")
plt.title("Stochastic Collapse of Wavepacket")
plt.legend()
plt.grid()
plt.show()
