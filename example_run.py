import numpy as np
from chaotic_sampler import chaotic_latent_sampler

# Example: Rössler latent data
data = np.loadtxt("rossler_trajectory.txt")

samples = chaotic_latent_sampler(
    data,
    B=100,
    B_fine=100,
    use_simulated_annealing=True,
    sa_iterations=20,
    num_rounds=1,
    mode="multi_seed"
)

print("Generated samples shape:", samples.shape)
