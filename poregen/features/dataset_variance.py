import numpy as np
import os


def diagonal_variance(datafolder: str,
                      savefolder: str,
                      nsamples: int,
                      tag: str = 'z'):
    # Ensure savefolder exists
    os.makedirs(savefolder, exist_ok=True)

    # Compute mean
    total_sum = 0
    for i in range(nsamples):
        data = np.load(f"{datafolder}/{i:05d}_{tag}.npy")
        total_sum += data
    mean = total_sum / nsamples
    scalar_mean = np.mean(mean)
    print(f"Scalar mean: {scalar_mean}")

    # Compute diagonal variance
    total_variance = 0
    for i in range(nsamples):
        data = np.load(f"{datafolder}/{i:05d}_{tag}.npy")
        total_variance += np.mean((data - mean) ** 2)
    variance = total_variance / nsamples
    print(f"Diagonal variance: {variance}")

    # Save result as a text file
    dict = {"scalar_mean": scalar_mean,
            "diagonal_variance": variance}
    with open(f"{savefolder}/latent_stats.txt", "w") as f:
        f.write(f"{dict}")
