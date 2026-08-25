import tempfile
from pathlib import Path

import numpy as np
import torch

from poregen.data.geomodel_datasets import (
    GeomodelVolumeDataset,
    as_nchw_volume,
    parse_well_xy,
    to_tricategorical,
    load_geomodel_array,
)
from poregen.data.geomodel_datamodule import GeomodelDataModule
from poregen.losses.slice_perceptual import SlicePerceptualLoss
from poregen.losses.well_enforcement import WellEnforcementLoss


def test_tricategorical_and_layout():
    raw = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    cats = to_tricategorical(raw)
    assert cats.tolist() == [0.0, 0.5, 1.0]

    grid = (4, 4, 2)
    nchw = np.zeros((3, 4, 4, 2), dtype=np.float32)
    out = as_nchw_volume(nchw, grid)
    assert out.shape == (3, 1, 4, 4, 2)

    n11 = np.zeros((3, 4, 4, 2, 1), dtype=np.float32)
    out = as_nchw_volume(n11, grid)
    assert out.shape == (3, 1, 4, 4, 2)


def test_parse_well_xy():
    wells = {"i1": [16, 64], "p4": [16, 16]}
    xy = parse_well_xy(wells, index_base=1)
    assert xy == [(15, 63), (15, 15)]


def test_well_enforcement_loss():
    volume_shape = (8, 8, 4)
    well_xy = [(2, 3), (5, 1)]
    loss_fn = WellEnforcementLoss(well_xy, volume_shape, reduction="mse")
    x = torch.zeros(2, 1, *volume_shape)
    assert float(loss_fn(x, x)) == 0.0

    recon = x.clone()
    recon[:, :, 2, 3, :] = 1.0
    well_err = float(loss_fn(recon, x))
    assert well_err > 0.0

    recon_off = x.clone()
    recon_off[:, :, 0, 0, :] = 1.0
    assert float(loss_fn(recon_off, x)) == 0.0


def test_slice_perceptual_identity():
    loss_fn = SlicePerceptualLoss(network_type="identity", slice_ratio=0.5)
    x = torch.rand(2, 1, 8, 8, 4)
    assert float(loss_fn(x, x, deterministic=True)) == 0.0
    y = x.clone()
    y[..., 0] = 1.0 - y[..., 0]
    assert float(loss_fn(y, x, deterministic=True)) > 0.0


def test_geomodel_datamodule_npy():
    volumes = np.zeros((10, 8, 8, 4), dtype=np.float32)
    volumes[::2] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geomodels.npy"
        np.save(path, volumes)
        loaded = load_geomodel_array(
            path, grid=[8, 8, 4], apply_thresholds=False
        )
        assert loaded.shape == (10, 1, 8, 8, 4)
        cfg = {
            "loader": "geomodel",
            "image_size": [8, 8, 4],
            "batch_size": 2,
            "num_workers": 0,
            "seed": 0,
            "apply_thresholds": False,
            "wells": {"i1": [1, 1]},
            "well_index_base": 1,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        }
        dm = GeomodelDataModule(path, cfg)
        dm.setup()
        assert len(dm.train_dataset) == 6
        assert len(dm.val_dataset) == 2
        assert len(dm.test_dataset) == 2
        batch = next(iter(dm.train_dataloader()))
        assert batch.shape == torch.Size([2, 1, 8, 8, 4])
        item = dm.train_dataset[0]
        assert item.shape == torch.Size([1, 8, 8, 4])
        assert dm.well_xy == [(0, 0)]


def test_volume_dataset():
    vols = np.zeros((5, 1, 4, 4, 2), dtype=np.float32)
    dataset = GeomodelVolumeDataset(vols)
    assert len(dataset) == 5
    assert dataset[1].shape == torch.Size([1, 4, 4, 2])


if __name__ == "__main__":
    test_tricategorical_and_layout()
    test_parse_well_xy()
    test_well_enforcement_loss()
    test_slice_perceptual_identity()
    test_geomodel_datamodule_npy()
    test_volume_dataset()
    print("All tests passed")
