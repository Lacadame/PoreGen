"""Lightning datamodule for stacked 3D facies geomodels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from .geomodel_datasets import (
    GeomodelVolumeDataset,
    load_geomodel_array,
    parse_well_xy,
)


def resolve_geomodel_path(
    path: str | Path,
    repo_root: Path | None = None,
) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if repo_root is not None:
        candidates.append(repo_root / path)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


class GeomodelDataModule(L.LightningDataModule):
    """Train/val/test split over full 128x128x32 geomodel volumes."""

    def __init__(
        self,
        data_path: str | Path = "",
        cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.cfg = cfg or {}
        if data_path == "":
            data_path = self.cfg.get("path", "")
        if data_path == "":
            raise ValueError("data_path or cfg.path must be provided")
        self.data_path = resolve_geomodel_path(data_path)
        self.well_xy: list[tuple[int, int]] = []

    def setup(self, stage=None):
        grid = self._grid()
        volumes = load_geomodel_array(
            self.data_path,
            grid=grid,
            h5_key=self.cfg.get("h5_key", "data"),
            thresh_low=self.cfg.get("thresh_low", 0.25),
            thresh_high=self.cfg.get("thresh_high", 0.80),
            apply_thresholds=self.cfg.get("apply_thresholds", True),
        )
        seed = int(self.cfg.get("seed", 0))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(volumes.shape[0])
        volumes = volumes[perm]

        n_train, n_val, _ = self._split_counts(volumes.shape[0])
        train = volumes[:n_train]
        val = volumes[n_train:n_train + n_val]
        test = volumes[n_train + n_val:]

        self.train_dataset = GeomodelVolumeDataset(train)
        self.val_dataset = GeomodelVolumeDataset(val)
        self.test_dataset = GeomodelVolumeDataset(test)
        self.well_xy = parse_well_xy(
            self.cfg.get("wells", {}),
            index_base=int(self.cfg.get("well_index_base", 1)),
        )

    def _grid(self) -> list[int]:
        image_size = self.cfg.get("image_size", [128, 128, 32])
        if isinstance(image_size, int):
            return [image_size] * 3
        return [int(v) for v in image_size]

    def _split_counts(self, n_total: int) -> tuple[int, int, int]:
        split = self.cfg.get("split", {})
        if "train_end" in split:
            n_train = int(split["train_end"])
            n_val = int(split.get("val_end", n_train)) - n_train
            n_test = n_total - n_train - n_val
            return n_train, n_val, n_test
        train_frac = float(split.get("train", 0.8))
        val_frac = float(split.get("val", 0.1))
        n_train = int(round(train_frac * n_total))
        n_val = int(round(val_frac * n_total))
        n_test = n_total - n_train - n_val
        return n_train, n_val, n_test

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.get("batch_size", 8),
            shuffle=shuffle,
            num_workers=self.cfg.get("num_workers", 0),
            pin_memory=self.cfg.get("pin_memory", True),
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
