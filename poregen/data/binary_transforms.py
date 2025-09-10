from typing import Literal

import torch
import torchvision.transforms.v2 as transforms


class RandomDimensionFlip(torch.nn.Module):
    def __init__(self, dim: int, p: float = 0.5):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.p < torch.rand(1):
            return img
        else:
            return img.flip(self.dim)


def get_mirror_transforms(dimension: Literal[2, 3] = 2) -> torch.nn.Module:
    if dimension == 2:
        return transforms.Compose([
            RandomDimensionFlip(dim=-1, p=0.5),
            RandomDimensionFlip(dim=-2, p=0.5),
        ])
    elif dimension == 3:
        return transforms.Compose([
            RandomDimensionFlip(dim=-1, p=0.5),
            RandomDimensionFlip(dim=-2, p=0.5),
            RandomDimensionFlip(dim=-3, p=0.5),
        ])
    else:
        raise ValueError(f"Dimension {dimension} not supported")


def get_plane_mirror_transforms(dimension: Literal[2, 3] = 2) -> torch.nn.Module:
    if dimension == 2:
        return transforms.Compose([
            RandomDimensionFlip(dim=-1, p=0.5),
            RandomDimensionFlip(dim=-2, p=0.5),
        ])
    elif dimension == 3:
        return transforms.Compose([
            RandomDimensionFlip(dim=-2, p=0.5),
            RandomDimensionFlip(dim=-3, p=0.5),
        ])
    else:
        raise ValueError(f"Dimension {dimension} not supported")
