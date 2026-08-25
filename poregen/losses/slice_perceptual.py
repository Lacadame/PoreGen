"""Orthogonal-slice perceptual loss for 3D geomodels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_backbone(network_type: str) -> nn.Module:
    network_type = network_type.lower()
    if network_type in {"identity", "none"}:
        return nn.Identity()
    try:
        from torchvision.models import (
            resnet18,
            resnet50,
            squeezenet1_1,
            ResNet18_Weights,
            ResNet50_Weights,
            SqueezeNet1_1_Weights,
        )
    except ImportError as exc:
        raise ImportError(
            "SlicePerceptualLoss requires torchvision."
        ) from exc
    if network_type in {"resnet", "resnet18"}:
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(net.children())[:-1])
    if network_type == "resnet50":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(net.children())[:-1])
    if network_type in {"squeeze", "squeezenet"}:
        net = squeezenet1_1(
            weights=SqueezeNet1_1_Weights.IMAGENET1K_V1
        )
        return net.features
    raise ValueError(
        f"Unknown perceptual network '{network_type}'. "
        "Use resnet18, resnet50, squeeze, or identity."
    )


class SlicePerceptualLoss(nn.Module):
    """Paper Eq. (5)-(6): ResNet features on random orthogonal slices.

    For each axis, a fraction ``slice_ratio`` of 2D slices is compared
    in a frozen ImageNet backbone. The three axis terms are summed.
    """

    def __init__(
        self,
        network_type: str = "resnet18",
        slice_ratio: float = 0.2,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        if not 0.0 < slice_ratio <= 1.0:
            raise ValueError("slice_ratio must be in (0, 1].")
        self.slice_ratio = slice_ratio
        self.backbone = backbone if backbone is not None else (
            _build_backbone(network_type)
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        reconstructions: torch.Tensor,
        inputs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if reconstructions.shape != inputs.shape:
            raise ValueError(
                "reconstruction and input shapes must match, got "
                f"{tuple(reconstructions.shape)} vs "
                f"{tuple(inputs.shape)}"
            )
        if reconstructions.ndim != 5:
            raise ValueError(
                "Expected (B, C, X, Y, Z) volumes, got "
                f"{tuple(reconstructions.shape)}"
            )
        total = reconstructions.new_zeros(())
        for axis in range(3):
            total = total + self._axis_loss(
                reconstructions, inputs, axis, deterministic
            )
        return total

    def _axis_loss(
        self,
        reconstructions: torch.Tensor,
        inputs: torch.Tensor,
        axis: int,
        deterministic: bool,
    ) -> torch.Tensor:
        length = reconstructions.shape[axis + 2]
        n_sub = max(1, int(length * self.slice_ratio))
        indices = self._select_indices(
            length, n_sub, reconstructions.device, deterministic
        )
        rec_slices = self._extract_slices(reconstructions, axis, indices)
        inp_slices = self._extract_slices(inputs, axis, indices)
        rec_feat = self.backbone(self._prepare(rec_slices))
        inp_feat = self.backbone(self._prepare(inp_slices))
        return F.mse_loss(rec_feat, inp_feat)

    def _select_indices(
        self,
        length: int,
        n_sub: int,
        device: torch.device,
        deterministic: bool,
    ) -> torch.Tensor:
        if deterministic:
            idx = torch.linspace(
                0, length - 1, n_sub, device=device
            ).round().long()
            return torch.unique(idx)
        return torch.randperm(length, device=device)[:n_sub]

    @staticmethod
    def _extract_slices(
        volume: torch.Tensor,
        axis: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        dim = axis + 2
        slices = volume.index_select(dim, indices)
        slices = slices.movedim(dim, 1)
        batch, n_slice, channels, height, width = slices.shape
        return slices.reshape(batch * n_slice, channels, height, width)

    def _prepare(self, slices: torch.Tensor) -> torch.Tensor:
        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)
        elif slices.shape[1] != 3:
            raise ValueError(
                "Perceptual loss expects 1 or 3 channels, got "
                f"{slices.shape[1]}."
            )
        return (slices - self.mean) / self.std
