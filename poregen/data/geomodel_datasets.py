"""Load 3D facies geomodels (Di Federico & Durlofsky, 2025)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


GEOMODEL_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1Z9xHuhkhXijOgxtGLFAhOX3yhPgfx-bd"
)


def to_tricategorical(
    array: np.ndarray,
    thresh_low: float = 0.25,
    thresh_high: float = 0.80,
) -> np.ndarray:
    """Map continuous values in [0, 1] to {0, 0.5, 1} facies."""
    out = np.empty_like(array, dtype=np.float32)
    out[array < thresh_low] = 0.0
    mid = (array >= thresh_low) & (array <= thresh_high)
    out[mid] = 0.5
    out[array > thresh_high] = 1.0
    return out


def maybe_scale_to_unit_interval(array: np.ndarray) -> np.ndarray:
    """Divide by 255 when the file is stored as 0-255 (or uint8)."""
    if np.issubdtype(array.dtype, np.integer) or array.max() > 1.5:
        return array.astype(np.float32) / 255.0
    return array.astype(np.float32)


def as_nchw_volume(
    array: np.ndarray,
    grid: Sequence[int],
) -> np.ndarray:
    """Return volumes as (N, 1, nx, ny, nz)."""
    nx, ny, nz = [int(v) for v in grid]
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim == 4:
        if tuple(array.shape[1:]) == (nx, ny, nz):
            return array[:, None, ...].astype(np.float32)
        raise ValueError(
            f"Expected (N, {nx}, {ny}, {nz}), got {array.shape}."
        )
    if array.ndim == 5:
        if array.shape[1] == 1 and tuple(array.shape[2:]) == (nx, ny, nz):
            return array.astype(np.float32)
        if array.shape[-1] == 1 and tuple(array.shape[1:4]) == (
                nx, ny, nz):
            return np.moveaxis(array, -1, 1).astype(np.float32)
        raise ValueError(
            f"Expected channel axis 1 or 4 with grid "
            f"{(nx, ny, nz)}, got {array.shape}."
        )
    raise ValueError(f"Unsupported geomodel array shape {array.shape}.")


def load_geomodel_array(
    path: str | Path,
    grid: Sequence[int],
    h5_key: str = "data",
    thresh_low: float = 0.25,
    thresh_high: float = 0.80,
    apply_thresholds: bool = True,
) -> np.ndarray:
    """Load an .h5 or .npy geomodel stack as (N, 1, nx, ny, nz)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Geomodel file not found: {path}\n"
            f"Download the Petrel .h5 from {GEOMODEL_DRIVE_URL}"
        )
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        array = _load_h5(path, h5_key)
    elif suffix == ".npy":
        array = np.load(path)
    else:
        raise ValueError(
            f"Unsupported geomodel file type '{path.suffix}'. "
            "Use .h5/.hdf5 or .npy."
        )
    array = maybe_scale_to_unit_interval(array)
    if apply_thresholds:
        array = to_tricategorical(array, thresh_low, thresh_high)
    return as_nchw_volume(array, grid)


def _load_h5(path: Path, h5_key: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading .h5 geomodels requires h5py. "
            "Install it with `pip install h5py`."
        ) from exc
    with h5py.File(path, "r") as handle:
        if h5_key not in handle:
            keys = list(handle.keys())
            raise KeyError(
                f"HDF5 key '{h5_key}' not found in {path}. "
                f"Available keys: {keys}"
            )
        return np.array(handle[h5_key])


def parse_well_xy(
    wells: dict | list,
    index_base: int = 1,
) -> list[tuple[int, int]]:
    """Parse well (x, y) locations and convert to 0-based indices."""
    if isinstance(wells, dict):
        pairs = list(wells.values())
    else:
        pairs = list(wells)
    xy = []
    for pair in pairs:
        if isinstance(pair, dict):
            pair = pair["xy"] if "xy" in pair else pair["location"]
        x, y = int(pair[0]), int(pair[1])
        xy.append((x - index_base, y - index_base))
    return xy


class GeomodelVolumeDataset(Dataset):
    """One full 3D geomodel per item, channel-first."""

    def __init__(self, volumes: np.ndarray | torch.Tensor):
        if isinstance(volumes, np.ndarray):
            volumes = torch.from_numpy(volumes)
        if volumes.ndim != 5:
            raise ValueError(
                f"Expected (N, 1, nx, ny, nz), got {tuple(volumes.shape)}."
            )
        self.volumes = volumes.float().contiguous()

    def __len__(self) -> int:
        return int(self.volumes.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.volumes[idx]
