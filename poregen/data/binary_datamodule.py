from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path
from .binary_datasets import (VoxelToSlicesDataset, VoxelToSubvoxelDataset,
                              SequenceOfVoxelsToSlicesDataset, SequenceOfVoxelsToSubvoxelDataset,
                              load_binary_from_eleven_sandstones, load_porespy_generated)
from poregen.features import feature_extractors


def get_binary_datamodule(data_path, cfg):  # noqa: C901
    class BinaryVoxelDataModule(L.LightningDataModule):
        def __init__(self, data_path, cfg):
            super().__init__()
            self.data_path = data_path
            self.cfg = cfg

        def setup(self, stage=None):
            voxels = self.load_voxels()

            # Choose dataset class based on dimension and data type
            if isinstance(voxels, list):
                dataset_class = (SequenceOfVoxelsToSubvoxelDataset
                                 if self.cfg['dimension'] == 3
                                 else SequenceOfVoxelsToSlicesDataset)
            else:
                dataset_class = (VoxelToSubvoxelDataset
                                 if self.cfg['dimension'] == 3
                                 else VoxelToSlicesDataset)

            # Prepare feature extractor
            feature_extractor = self.get_feature_extractor()

            # Prepare dataset arguments
            dataset_args = {
                'subslice': self.cfg['image_size'],
                'voxel_downscale_factor': self.cfg['voxel_downscale_factor'],
                'feature_extractor': feature_extractor,
                'center': self.cfg.get('center', False),
                'invert': self.cfg.get('invert', False),
            }

            self.train_dataset = dataset_class(voxels,
                                               dataset_size=self.cfg['training_dataset_size'],
                                               **dataset_args)

            self.val_dataset = dataset_class(voxels,
                                             dataset_size=self.cfg['validation_dataset_size'],
                                             **dataset_args)

        def load_voxels(self):
            loader = self.cfg.get('loader', 'eleven_sandstones')
            if isinstance(self.data_path, (str, Path)):
                if loader == 'eleven_sandstones':
                    return load_binary_from_eleven_sandstones(self.data_path)
                elif loader == 'porespy':
                    return load_porespy_generated(self.data_path)
            elif isinstance(self.data_path, list):
                return [self.load_single_voxel(path) for path in self.data_path]
            raise ValueError(f"Unsupported data path or loader: {self.data_path}, {loader}")

        def load_single_voxel(self, path):
            loader = self.cfg.get('loader', 'eleven_sandstones')
            if loader == 'eleven_sandstones':
                return load_binary_from_eleven_sandstones(path)
            elif loader == 'porespy':
                return load_porespy_generated(path)
            raise ValueError(f"Unsupported loader: {loader}")

        def get_feature_extractor(self):
            feature_config = self.cfg.get('feature_extractor')
            if not feature_config:
                return None
            if isinstance(feature_config, str):
                extractor_kwargs = self.cfg.get('feature_extractor_kwargs', {})
                return feature_extractors.make_feature_extractor(
                    feature_config,
                    **extractor_kwargs
                )
            elif isinstance(feature_config, list):
                extractor_names = feature_config
                extractor_kwargs = self.cfg.get('feature_extractor_kwargs', {})
                return feature_extractors.make_composite_feature_extractor(
                    extractor_names, extractor_kwargs)
            else:
                raise ValueError(f"Unsupported feature extractor configuration: {feature_config}")

        def train_dataloader(self):
            return DataLoader(self.train_dataset,
                              batch_size=self.cfg['batch_size'],
                              shuffle=True,
                              num_workers=self.cfg['num_workers'])

        def val_dataloader(self):
            return DataLoader(self.val_dataset,
                              batch_size=self.cfg['batch_size'],
                              shuffle=False,
                              num_workers=self.cfg['num_workers'])

    return BinaryVoxelDataModule(data_path, cfg)
