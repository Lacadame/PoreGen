# flake8: noqa

from .binary_datasets import (load_binary_from_eleven_sandstones,
                              load_porespy_generated,
                              get_standard_binary_transforms,
                              VoxelToSlicesDataset,
                              SequenceOfVoxelsToSlicesDataset,
                              VoxelToSubvoxelDataset,
                              SequenceOfVoxelsToSubvoxelDataset,
                              VoxelToSubvoxelSequentialDataset)
from .geomodel_datasets import (
    GeomodelVolumeDataset,
    load_geomodel_array,
    parse_well_xy,
    to_tricategorical,
)
from .geomodel_datamodule import GeomodelDataModule


def get_binary_datamodule(*args, **kwargs):
    from .binary_datamodule import get_binary_datamodule as _get
    return _get(*args, **kwargs)


def get_datamodule(data_path, cfg, stride=None):
    loader = cfg.get("loader", "eleven_sandstones")
    if loader in {"geomodel", "geomodel_h5", "geomodel_npy"}:
        return GeomodelDataModule(data_path, cfg)
    from .binary_datamodule import BinaryVoxelDataModule
    return BinaryVoxelDataModule(data_path, cfg, stride)
