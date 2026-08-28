"""Mean-L1 recon + well losses matching guidodf09/ldm_3d_geomodel.

Their trainer uses ``nn.L1Loss()`` over the full volume and
``F.l1_loss`` over well cells, both with default mean reduction, and
``hd_weight=0.01``. DiffSci's VAE recon is a *sum* of squared errors, so
the same 0.01 on :class:`WellEnforcementLoss` is not on that scale.

This module is a separate objective: both terms are mean L1 (O(1)), so
``well_weight=0.01`` has the same relative weight as their ``hd: 0.01``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class GeomodelMeanL1Loss(nn.Module):
    """Volume mean L1 plus optional well-column mean L1.

    Parameters
    ----------
    well_xy:
        Well (x, y) indices in 0-based grid coordinates. Empty disables
        the well term (returns 0).
    volume_shape:
        ``(nx, ny, nz)``.
    """

    def __init__(
        self,
        well_xy: Sequence[tuple[int, int]] | None,
        volume_shape: Sequence[int],
    ):
        super().__init__()
        nx, ny, nz = [int(v) for v in volume_shape]
        mask = torch.zeros(1, 1, nx, ny, nz)
        well_xy = list(well_xy or [])
        for x, y in well_xy:
            if not (0 <= x < nx and 0 <= y < ny):
                raise ValueError(
                    f"Well ({x}, {y}) is outside grid {(nx, ny, nz)}."
                )
            mask[:, :, x, y, :] = 1.0
        self.register_buffer("mask", mask)
        self.n_hard = int(mask.sum().item())

    def forward(
        self,
        reconstructions: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        err = (reconstructions - inputs).abs()
        rec = err.mean()
        if self.n_hard == 0:
            well = err.new_zeros(())
            return rec, well
        mask = self.mask.to(dtype=err.dtype)
        # Mean over batch, channel, and well cells (F.l1_loss default).
        well_count = mask.sum() * err.shape[0] * err.shape[1]
        well = (err * mask).sum() / well_count.clamp(min=1.0)
        return rec, well
