import warnings
from typing import Callable, List, Union, Tuple, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

from . import binary_transforms


TransformTypesStr = Literal['mirror', 'plane_mirror']


def get_standard_binary_transforms(dimension: Literal[2, 3] = 2) -> torch.nn.Module:
    return binary_transforms.get_mirror_transforms(dimension)


def reduce_tensor(tensor: torch.Tensor,
                  factor: int,
                  mode: str = 'avg') -> torch.Tensor:
    if len(tensor.shape) not in (2, 3):
        raise ValueError("Tensor must be 2D or 3D")

    pool_func = getattr(torch.nn.functional, f'{mode}_pool{len(tensor.shape)}d')
    return pool_func(tensor.unsqueeze(0).float(), factor, factor).squeeze(0)


def load_binary_from_eleven_sandstones(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        unshaped_voxel = np.fromfile(f, dtype=np.uint8)

    size = int(round(unshaped_voxel.shape[0] ** (1 / 3)))
    if size**3 != unshaped_voxel.shape[0]:
        raise ValueError('Voxel is not a cube.')

    return unshaped_voxel.reshape((size, size, size))


def load_porespy_generated(path: str) -> np.ndarray:
    return np.load(path).astype(np.uint8)


class BaseVoxelDataset(Dataset):
    output_dim: Literal[2, 3] = 2

    def __init__(
        self,
        voxels: Union[np.ndarray, List[np.ndarray]],
        subslice: Union[int, List[int]] = 64,
        dataset_size: int = 34560,
        voxel_downscale_factor: int = 1,
        transform: None | bool | torch.nn.Module | TransformTypesStr = False,
        feature_extractor: Callable = None,
        center: bool = False,
        invert: bool = False,
        image_size: Union[int, List[int]] = None,
        pool_mode: str = 'avg',
        return_as_dict: bool = False
    ):  
        if image_size is not None:
            warnings.warn("'image_size' is deprecated, use 'subslice' instead",
                          DeprecationWarning)
            subslice = image_size

        self.voxels = [torch.tensor(voxel, dtype=torch.float32)
                       for voxel in (voxels if isinstance(voxels, list)
                                     else [voxels])]

        if voxel_downscale_factor > 1:
            self.voxels = [reduce_tensor(voxel, voxel_downscale_factor,
                                         pool_mode)
                           for voxel in self.voxels]

        self.subslice = ([subslice] * 3 if isinstance(subslice, int)
                         else list(subslice))
        assert len(self.subslice) == (3 if self.is_3d() else 2), (
            "Subslice dimensions must match voxel dimensions")

        self.dataset_size = dataset_size
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.center = center
        self.invert = invert
        self.return_as_dict = return_as_dict
        self.transform = self.set_transform(transform)

    def set_transform(self, transform: bool | torch.nn.Module | TransformTypesStr) -> torch.nn.Module:
        if transform is None:
            return None
        elif isinstance(transform, bool):
            warnings.warn("type 'bool' for 'transform' is deprecated, use a torch.nn.Module or a string")
            if transform:
                warnings.warn("transform=True will be read as transform='standard'")
                transform = binary_transforms.get_mirror_transforms(self.output_dim)
            else:
                warnings.warn("transform=False will be read as transform=None")
                transform = None
        elif isinstance(transform, str):
            if transform == 'mirror':
                transform = binary_transforms.get_mirror_transforms(self.output_dim)
            elif transform == "plane_mirror":
                transform = binary_transforms.get_plane_mirror_transforms(self.output_dim)
            else:
                raise ValueError(f"Invalid transform type: {transform}")
        return transform

    def is_3d(self) -> bool:
        return len(self.voxels[0].shape) == 3

    def get_random_voxel(self) -> torch.Tensor:
        return self.voxels[np.random.randint(len(self.voxels))]

    def __len__(self) -> int:
        return self.dataset_size

    def process_crop(self, crop: torch.Tensor) -> torch.Tensor:
        if self.transform:
            crop = self.transform(crop)
        if self.invert:
            crop = 1 - crop
        if self.center:
            crop = 2 * crop - 1
        return crop

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor,
                                                                 torch.Tensor]]:
        raise NotImplementedError


class VoxelToSlicesDataset(BaseVoxelDataset):
    output_dim: Literal[2, 3] = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cropper = transforms.RandomCrop(self.subslice[:2])

    def __getitem__(self, idx: int) -> Union[torch.Tensor,
                                             Tuple[torch.Tensor, torch.Tensor]]:
        voxel = self.get_random_voxel()
        slice_idx = np.random.randint(voxel.shape[0])
        x = voxel[slice_idx].unsqueeze(0)
        x = self.cropper(x)
        x = self.process_crop(x)

        if self.feature_extractor:
            y = self.feature_extractor(x)
            if self.return_as_dict:
                return {'x': x, 'y': y}
            else:
                return x, y
        elif self.return_as_dict:
            return {'x': x}
        else:
            return x


class VoxelToSubvoxelDataset(BaseVoxelDataset):
    output_dim: Literal[2, 3] = 2

    def __getitem__(self, idx: int) -> Union[torch.Tensor,
                                             Tuple[torch.Tensor, torch.Tensor]]:
        voxel = self.get_random_voxel()
        starts = [np.random.randint(0, dim - sub + 1)
                  for dim, sub in zip(voxel.shape, self.subslice)]
        slices = tuple(slice(start, start + sub)
                       for start, sub in zip(starts, self.subslice))
        crop = voxel[slices].unsqueeze(0)
        crop = self.process_crop(crop)

        if self.feature_extractor:
            y = self.feature_extractor(crop)
            if self.return_as_dict:
                return {'x': crop, 'y': y}
            else:
                return crop, y
        elif self.return_as_dict:
            return {'x': crop}
        else:
            return crop


class VoxelToSubvoxelSequentialDataset(BaseVoxelDataset):
    """
    Sequential, stride-controlled sub-voxel sampler.
    Returns None if idx ≥ total number of admissible sub-voxels.
    """
    output_dim: Literal[2, 3] = 2

    def __init__(self, stride: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride

        self.starts_per_axis = [
            list(range(0, dim - sub + 1, stride))
            for dim, sub in zip(self.voxels[0].shape, self.subslice)
        ]
        print('starts_per_axis:', self.starts_per_axis)
        self.per_voxel = np.prod([len(ax) for ax in self.starts_per_axis])
        self.dataset_size = self.per_voxel * len(self.voxels)
        print('Dataset_size:', self.dataset_size)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if idx >= self.dataset_size:
            return None

        # Decode voxel and coordinate indices
        voxel_idx = idx // self.per_voxel
        within_voxel_idx = idx % self.per_voxel
        voxel = self.voxels[voxel_idx]

        # Calculate grid dimensions
        nx = len(self.starts_per_axis[-1])  # x-axis
        ny = len(self.starts_per_axis[-2])  # y-axis

        # Decode 3D coordinates from linear index
        z_lin = within_voxel_idx // (ny * nx)
        rem = within_voxel_idx % (ny * nx)
        y_lin = rem // nx
        x_lin = rem % nx

        # Get starting coordinates for each axis
        starts = (
            self.starts_per_axis[0][z_lin],
            self.starts_per_axis[1][y_lin],
            self.starts_per_axis[2][x_lin],
        )

        # Extract and process the subvoxel
        slices = tuple(slice(s, s + sub) for s, sub in zip(starts, self.subslice))
        crop = voxel[slices].unsqueeze(0)
        crop = self.process_crop(crop)

        if self.feature_extractor is not None:
            return crop, self.feature_extractor(crop)
        return crop


# Aliases for backward compatibility
SequenceOfVoxelsToSlicesDataset = VoxelToSlicesDataset
SequenceOfVoxelsToSubvoxelDataset = VoxelToSubvoxelDataset
