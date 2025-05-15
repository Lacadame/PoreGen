from typing import Any
import os

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import torchmetrics
from torch import Tensor
from jaxtyping import Float
import porespy

from poregen.utils import inverse_cdf_histogram


def nearest_neighbour(generated_samples, dataset, vae_model=None):
    """
    Find the nearest neighbour of generated samples in some dataset.
    """
    nearest = []
    for sample in generated_samples:
        distance, x_train = find_nearest(sample, dataset, vae_model)
        nearest.append([distance, x_train])
    return nearest


def find_nearest(sample, dataset, vae_model=None):
    """
    Find the nearest neighbour of a sample in some dataset.
    """
    n = dataset.size
    for i in range(n):
        x_train = dataset.get_item(i)
        if vae_model is not None:
            sample = vae_model.encode(sample)
            x_train = vae_model.encode(x_train)
        distance = torch.linalg.norm(sample - x_train)
        if i == 0:
            nearest = distance
            x_train_nearest = x_train
        elif distance < nearest:
            nearest = distance
            x_train_nearest = x_train
    return nearest, x_train_nearest
