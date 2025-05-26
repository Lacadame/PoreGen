from typing import Any
import os
import time

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import torchmetrics
from torch import Tensor
from jaxtyping import Float
import porespy

from poregen.utils import inverse_cdf_histogram


def nearest_neighbour(generated_samples, dataset, vae_model=None, max_samples=None, batch_size=32, device_id=7):
    """
    Find the nearest neighbour of generated samples in some dataset using batched processing.
    Memory efficient version that processes both generated and dataset samples in batches.
    
    Args:
        generated_samples: Array of generated samples
        dataset: Dataset to search in
        vae_model: Optional VAE model for latent space comparison
        max_samples: Maximum number of samples to process
        batch_size: Maximum number of samples to keep in memory at once
        device_id: GPU device ID to use (0-7). If None, uses CPU.
        
    Returns:
        List of tuples containing (minimum_distance, nearest_neighbor_index) for each generated sample
    """
    # Check if CUDA is available and set device
    if device_id is None:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            print(f"Warning: GPU device {device_id} requested but CUDA is not available. Falling back to CPU.")
            device = torch.device('cpu')
        elif device_id >= torch.cuda.device_count():
            print(f"Warning: GPU device {device_id} requested but only {torch.cuda.device_count()} GPUs available. "
                  f"Falling back to GPU 0.")
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{device_id}')
    
    print(f"Using device: {device}")
    
    if max_samples is not None:
        generated_samples = generated_samples[:max_samples]
    
    # Convert to tensor and move to device
    generated_samples = torch.tensor(generated_samples).float().to(device)
    n_samples = len(generated_samples)
    dataset_size = dataset.dataset_size
    dataset_batch_size = max(1, batch_size // 2)  # Use half the batch size for dataset samples
    
    print(f"\nProcessing {n_samples} generated samples against {dataset_size} dataset samples")
    print(f"Using batch sizes: generated={batch_size}, dataset={dataset_batch_size}")
    
    nearest = []
    
    # Process generated samples in batches
    for i in range(0, n_samples, batch_size):
        batch_start_time = time.time()
        gen_batch = generated_samples[i:i + batch_size]
        
        # VAE encoding time for generated batch
        vae_start = time.time()
        if vae_model is not None:
            with torch.no_grad():
                gen_batch = vae_model.encode(gen_batch)
        vae_time = time.time() - vae_start
        
        # Pre-compute squared norm of generated samples
        gen_norm = torch.sum(gen_batch.view(gen_batch.size(0), -1) ** 2, dim=1, keepdim=True)
        
        # Initialize storage for this batch of generated samples
        batch_distances = []
        batch_offsets = []  # Store the starting index of each batch
        
        # Process dataset in batches
        for j in range(0, dataset_size, dataset_batch_size):
            # Load and encode dataset batch
            load_start = time.time()
            dataset_batch = []
            for k in range(j, min(j + dataset_batch_size, dataset_size)):
                data_load_start = time.time()
                x_data = dataset[k].float().to(device)  # Move to device
                data_load_time = time.time() - data_load_start
                
                if vae_model is not None:
                    with torch.no_grad():
                        x_data = vae_model.encode(x_data)
                dataset_batch.append(x_data)
            dataset_batch = torch.stack(dataset_batch)
            load_time = time.time() - load_start
            
            # Calculate distances more efficiently
            dist_start = time.time()
            
            # Reshape for efficient computation
            gen_flat = gen_batch.view(gen_batch.size(0), -1)
            dataset_flat = dataset_batch.view(dataset_batch.size(0), -1)
            
            # Compute squared norm of dataset samples
            dataset_norm = torch.sum(dataset_flat ** 2, dim=1, keepdim=True)
            
            # Compute distances using the expanded form of ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            # This avoids the need for broadcasting large tensors
            distances = gen_norm + dataset_norm.t() - 2 * torch.mm(gen_flat, dataset_flat.t())
            distances = torch.sqrt(torch.clamp(distances, min=0))  # Ensure non-negative due to numerical precision
            
            dist_time = time.time() - dist_start
            
            # Store distances and batch offset for this dataset batch
            batch_distances.append(distances)
            batch_offsets.append(j)
            
            # Memory management
            del dataset_batch, dataset_flat, dataset_norm
            torch.cuda.empty_cache()
            
            # Print progress with timing information
            print(f"\rDataset batch {j//dataset_batch_size + 1}/{(dataset_size + dataset_batch_size - 1)//dataset_batch_size}: "
                  f"Data load: {data_load_time:.3f}s/sample, "
                  f"Total load: {load_time:.2f}s, "
                  f"Distance calc: {dist_time:.2f}s", end="")
        
        print()  # New line after dataset progress
        
        # Combine all distances for this generated batch
        all_distances = torch.cat(batch_distances, dim=1)
        
        # Find nearest neighbor for each sample in generated batch
        min_distances, min_indices = torch.min(all_distances, dim=1)
        
        # Get the corresponding nearest neighbor indices
        for dist, idx in zip(min_distances, min_indices):
            batch_idx = idx // dataset_batch_size
            sample_idx = idx % dataset_batch_size
            nearest_idx = batch_offsets[batch_idx] + sample_idx
            nearest.append([dist.item(), nearest_idx])
        
        batch_time = time.time() - batch_start_time
        print(f"Generated batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: "
              f"VAE time: {vae_time:.2f}s, "
              f"Total batch time: {batch_time:.2f}s")
        
        # Clear memory
        del gen_batch, gen_flat, gen_norm, all_distances, batch_distances, batch_offsets
        torch.cuda.empty_cache()
    
    return nearest
