import warnings
from typing import Callable, List, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


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


def get_standard_binary_transforms() -> transforms.Transform:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])


class BaseVoxelDataset(Dataset):
    def __init__(
        self,
        voxels: Union[np.ndarray, List[np.ndarray]],
        subslice: Union[int, List[int]] = 64,
        dataset_size: int = 34560,
        voxel_downscale_factor: int = 1,
        transform: bool = False,
        feature_extractor: Callable = None,
        center: bool = False,
        invert: bool = False,
        image_size: Union[int, List[int]] = None,
        pool_mode: str = 'avg'
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

    def is_3d(self) -> bool:
        return len(self.voxels[0].shape) == 3

    def get_random_voxel(self) -> torch.Tensor:
        return self.voxels[np.random.randint(len(self.voxels))]

    def __len__(self) -> int:
        return self.dataset_size

    def process_crop(self, crop: torch.Tensor) -> torch.Tensor:
        if self.transform:
            transform = get_standard_binary_transforms()
            crop = transform(crop)
        if self.invert:
            crop = 1 - crop
        if self.center:
            crop = 2 * crop - 1
        return crop

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor,
                                                                 torch.Tensor]]:
        raise NotImplementedError


class VoxelToSlicesDataset(BaseVoxelDataset):
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
            return x, y
        return x


class VoxelToSubvoxelDataset(BaseVoxelDataset):
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
            return crop, y
        return crop


# Aliases for backward compatibility
SequenceOfVoxelsToSlicesDataset = VoxelToSlicesDataset
SequenceOfVoxelsToSubvoxelDataset = VoxelToSubvoxelDataset
