"""3D AutoencoderKL with geomodel perceptual and well losses."""

from __future__ import annotations

from typing import Sequence

import torch

from diffsci.models.nets.autoencoderldm3d import AutoencoderKL

from poregen.losses.geomodel_mean_l1 import GeomodelMeanL1Loss
from poregen.losses.slice_perceptual import SlicePerceptualLoss
from poregen.losses.well_enforcement import WellEnforcementLoss

_MEAN_L1_KEYS = {"mean_l1", "github", "l1"}


class GeomodelAutoencoderKL(AutoencoderKL):
    """DiffSci AutoencoderKL plus paper extra VAE loss terms.

    DiffSci already provides voxel reconstruction and KL. This wrapper
    adds:
    * orthogonal-slice perceptual loss (Eq. 5-6)
    * well-column hard-data loss L_h

    ``recon_loss='ldm'`` (default) keeps DiffSci's summed MSE recon plus
    :class:`WellEnforcementLoss`. ``recon_loss='mean_l1'`` switches to
    :class:`GeomodelMeanL1Loss` so recon and wells are both mean L1, matching
    https://github.com/guidodf09/ldm_3d_geomodel relative scales.
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim: int = 1,
        perceptual_weight: float = 1e-3,
        well_weight: float = 1e-2,
        well_xy: Sequence[tuple[int, int]] | None = None,
        volume_shape: Sequence[int] | None = None,
        perceptual_network: str = "resnet18",
        slice_ratio: float = 0.2,
        well_reduction: str = "mse",
        recon_loss: str = "ldm",
        **kwargs,
    ):
        super().__init__(
            ddconfig,
            lossconfig,
            distillconfig=kwargs.get("distillconfig"),
            embed_dim=embed_dim,
            ckpt_path=kwargs.get("ckpt_path"),
            ignore_keys=kwargs.get("ignore_keys", []),
            image_key=kwargs.get("image_key", "image"),
            colorize_nlabels=kwargs.get("colorize_nlabels"),
            monitor=kwargs.get("monitor"),
        )
        self.perceptual_weight = float(perceptual_weight)
        self.well_weight = float(well_weight)
        self.recon_loss = str(recon_loss).lower()
        self.use_mean_l1 = self.recon_loss in _MEAN_L1_KEYS
        if not self.use_mean_l1 and self.recon_loss not in {"ldm", "mse"}:
            raise ValueError(
                f"Unknown recon_loss={recon_loss!r}. Use 'ldm' or 'mean_l1'."
            )
        self.perceptual = None
        self.well_loss = None
        self.mean_l1_loss = None
        if self.perceptual_weight > 0:
            self.perceptual = SlicePerceptualLoss(
                network_type=perceptual_network,
                slice_ratio=slice_ratio,
            )
        if self.use_mean_l1:
            if volume_shape is None:
                raise ValueError(
                    "recon_loss='mean_l1' requires volume_shape."
                )
            if self.well_weight > 0 and not well_xy:
                raise ValueError(
                    "well_weight > 0 requires well_xy and volume_shape."
                )
            self.mean_l1_loss = GeomodelMeanL1Loss(
                well_xy=well_xy,
                volume_shape=volume_shape,
            )
        elif self.well_weight > 0:
            if not well_xy or volume_shape is None:
                raise ValueError(
                    "well_weight > 0 requires well_xy and volume_shape."
                )
            self.well_loss = WellEnforcementLoss(
                well_xy=well_xy,
                volume_shape=volume_shape,
                reduction=well_reduction,
            )

    @staticmethod
    def _kl_mean(posterior) -> torch.Tensor:
        """Sum KL over latent cells, mean over batch (GitHub KL_loss)."""
        kl = posterior.kl()
        if not torch.is_tensor(kl) or kl.ndim == 0:
            return kl
        return kl.reshape(kl.shape[0], -1).sum(dim=-1).mean()

    def _shared_step(self, batch, split: str):
        inputs = batch
        reconstructions, posterior = self(inputs)
        if self.use_mean_l1:
            rec, well = self.mean_l1_loss(reconstructions, inputs)
            kl = self._kl_mean(posterior)
            aeloss = rec + float(self.loss.kl_weight) * kl
            log_dict = {
                f"{split}/rec_loss": rec.detach(),
                f"{split}/kl_loss": kl.detach(),
                f"{split}/nll_loss": rec.detach(),
                f"{split}/total_loss": aeloss.detach(),
            }
            if self.well_weight > 0:
                aeloss = aeloss + self.well_weight * well
                log_dict[f"{split}/well_loss"] = well.detach()
        else:
            aeloss, log_dict = self.loss(
                inputs,
                reconstructions,
                posterior,
                self.global_step,
                last_layer=self.get_last_layer(),
                split=split,
            )
            if self.well_loss is not None and self.well_weight > 0:
                well = self.well_loss(reconstructions, inputs)
                aeloss = aeloss + self.well_weight * well
                log_dict[f"{split}/well_loss"] = well.detach()
        if self.perceptual is not None and self.perceptual_weight > 0:
            perc = self.perceptual(
                reconstructions,
                inputs,
                deterministic=(split != "train"),
            )
            aeloss = aeloss + self.perceptual_weight * perc
            log_dict[f"{split}/perc_loss"] = perc.detach()
        log_dict[f"{split}/total_loss"] = aeloss.detach()
        return aeloss, log_dict

    def training_step(self, batch, batch_idx):
        aeloss, log_dict = self._shared_step(batch, "train")
        self.log(
            "aeloss", aeloss, prog_bar=True, logger=True,
            on_step=True, on_epoch=True,
        )
        self.log_dict(
            log_dict, prog_bar=False, logger=True,
            on_step=True, on_epoch=False,
        )
        return aeloss

    def validation_step(self, batch, batch_idx):
        aeloss, log_dict = self._shared_step(batch, "val")
        rec_key = "val/rec_loss"
        monitor = log_dict[rec_key] if rec_key in log_dict else aeloss
        self.log("val_loss", monitor)
        self.log_dict(log_dict)
        return log_dict
