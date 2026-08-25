"""Well-column (hard-data) reconstruction loss."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class WellEnforcementLoss(nn.Module):
    """L2/L1 reconstruction error restricted to well columns.

    Matches L_h in Di Federico & Durlofsky (2025): the mean squared
    (or absolute) error between input and reconstruction at the N_h
    hard-data cells, i.e. the full vertical column at each well (x, y).
    """

    def __init__(
        self,
        well_xy: Sequence[tuple[int, int]],
        volume_shape: Sequence[int],
        reduction: str = "mse",
    ):
        super().__init__()
        if reduction not in {"mse", "l1", "mae"}:
            raise ValueError(
                f"Unsupported reduction '{reduction}'. "
                "Use 'mse' or 'l1'."
            )
        self.reduction = "l1" if reduction == "mae" else reduction
        nx, ny, nz = [int(v) for v in volume_shape]
        mask = torch.zeros(1, 1, nx, ny, nz)
        if not well_xy:
            raise ValueError("well_xy must contain at least one well.")
        for x, y in well_xy:
            if not (0 <= x < nx and 0 <= y < ny):
                raise ValueError(
                    f"Well ({x}, {y}) is outside grid "
                    f"{(nx, ny, nz)}."
                )
            mask[:, :, x, y, :] = 1.0
        self.register_buffer("mask", mask)
        self.n_hard = int(mask.sum().item())

    def forward(
        self,
        reconstructions: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.mask.to(dtype=reconstructions.dtype)
        if self.reduction == "mse":
            cell = (reconstructions - inputs) ** 2
        else:
            cell = (reconstructions - inputs).abs()
        return (cell * mask).sum() / mask.sum().clamp(min=1.0)
