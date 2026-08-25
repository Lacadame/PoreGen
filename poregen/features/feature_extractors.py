from typing import Callable, Any, Literal, Union, Dict, List
import functools
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch
import skimage.measure

from . import basicmetrics

KwargsType = dict[str, Any]


class InputType(Enum):
    """Supported input data types."""
    BINARY = "binary"
    EDT = "edt"  # Euclidean Distance Transform


class DataDimension(Enum):
    """Data dimensionality."""
    TWO_D = "2d"
    THREE_D = "3d"
    MIXED = "mixed"  # Can handle both 2D and 3D


class BaseExtractor(ABC):
    """Base class for all feature extractors."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this extractor."""
        pass

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """List of keys this extractor returns."""
        pass

    @property
    @abstractmethod
    def supported_dimensions(self) -> DataDimension:
        """Supported data dimensions."""
        pass

    @abstractmethod
    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from the input data."""
        pass

    def __call__(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.extract(data)


class PorosityExtractor(BaseExtractor):
    """Extract porosity from 2D slices or 3D volumes."""

    @property
    def name(self) -> str:
        return "porosity"

    @property
    def output_keys(self) -> List[str]:
        return ["porosity"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.MIXED

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        porosity = torch.tensor([(1 - data.numpy().mean())], dtype=torch.float)
        return {"porosity": porosity}


class SlicePorosityExtractor(BaseExtractor):
    """Extract porosity for each slice along the last dimension of a 3D volume."""

    @property
    def name(self) -> str:
        return "slice_porosity"

    @property
    def output_keys(self) -> List[str]:
        return ["slice", "porosity"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.THREE_D

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        porosity_extractor = PorosityExtractor()
        porosity = torch.tensor([
            porosity_extractor.extract(data[..., i])["porosity"]
            for i in range(data.shape[-1])
        ], dtype=torch.float)

        return {
            "slice": torch.arange(data.shape[-1]),
            "porosity": porosity
        }


class SubvolumePorosityExtractor(BaseExtractor):
    """Extract porosity along subvolumes of a partitioned 3D volume."""
    def __init__(self, size: int = 8):
        super().__init__()
        self.size = size

    @property
    def name(self) -> str:
        return "subvolume_porosity"

    @property
    def output_keys(self) -> List[str]:
        return ["slice", "porosity"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.THREE_D

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        porosity_extractor = PorosityExtractor()
        size = self.size
        subvolumes = torch.stack([data[..., i:i+size, j:j+size, k:k+size]
                                 for i in range(0, data.shape[-3], size)
                                 for j in range(0, data.shape[-2], size)
                                 for k in range(0, data.shape[-1], size)])
        porosity = torch.tensor([
                    porosity_extractor.extract(subvolumes[..., i, :, :, :, :])["porosity"]
                    for i in range(subvolumes.shape[-5])
        ], dtype=torch.float)

        return {
            "slice": torch.arange(subvolumes.shape[-5]),
            "porosity": porosity
        }


class EffectivePorosityExtractor(BaseExtractor):
    """Extract effective porosity (connected pore space)."""

    @property
    def name(self) -> str:
        return "effective_porosity"

    @property
    def output_keys(self) -> List[str]:
        return ["effective_porosity"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.MIXED

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        from . import filters

        volume = 1 - data[0].numpy()
        effective_porosity = filters.fill_blind_pores(volume).mean()
        return {"effective_porosity": torch.tensor([effective_porosity], dtype=torch.float)}


class TwoPointCorrelationExtractor(BaseExtractor):
    """Extract two-point correlation function."""

    def __init__(self, bins: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins

    @property
    def name(self) -> str:
        return "two_point_correlation"

    @property
    def output_keys(self) -> List[str]:
        return ["tpc_dist", "tpc_prob"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.MIXED

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        data = data[0].float()
        tpc_data = basicmetrics.two_point_correlation((1 - data).numpy(), bins=self.bins)
        dist = torch.tensor(tpc_data.distance, dtype=torch.float)
        prob = torch.tensor(tpc_data.probability_scaled, dtype=torch.float)
        prob = torch.nan_to_num(prob)
        return {"tpc_dist": dist, "tpc_prob": prob}


class PorosimetryExtractor(BaseExtractor):
    """Extract pore size distribution and moments."""

    def __init__(self, bins: int = 32, log: bool = False, maximum_momentum: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.log = log
        self.maximum_momentum = maximum_momentum

    @property
    def name(self) -> str:
        return "porosimetry"

    @property
    def output_keys(self) -> List[str]:
        return ["psd_centers", "psd_cdf", "psd_pdf", "log_momenta", "root_momenta", "standardized_momenta"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.MIXED

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        from . import porosimetry

        data = data[0].float()
        im = porosimetry.local_thickness((1 - data).numpy())
        psd_data = basicmetrics.pore_size_distribution(im, bins=self.bins, log=self.log)
        bin_centers = torch.tensor(psd_data.bin_centers.copy(), dtype=torch.float)
        cdf = torch.tensor(psd_data.cdf.copy(), dtype=torch.float)
        pdf = torch.tensor(psd_data.pdf.copy(), dtype=torch.float)

        raw_data = im.flatten()[im.flatten() > 0]
        data_size = raw_data.shape[0]
        momenta = []
        for i in range(self.maximum_momentum):
            momenta.append((raw_data**(i+1)).sum(axis=0)/data_size)
        log_momenta = torch.tensor(np.log(np.array(momenta)), dtype=torch.float)
        root_momenta = momenta**np.array([1/(i+1) for i in range(self.maximum_momentum)])
        root_momenta = torch.tensor(root_momenta, dtype=torch.float)

        standardized_momenta = self._calculate_standardized_momenta(raw_data, self.maximum_momentum)

        return {
            "psd_centers": bin_centers,
            "psd_cdf": cdf,
            "psd_pdf": pdf,
            "log_momenta": log_momenta,
            "root_momenta": root_momenta,
            "standardized_momenta": standardized_momenta
        }

    def _calculate_standardized_momenta(self, raw_data, maximum_momentum):
        """Calculate the standardized momenta."""
        standardized_momenta = []
        assert maximum_momentum > 0
        # First standardized momenta is just the mean
        mean = raw_data.mean()
        standardized_momenta.append(mean)
        if maximum_momentum > 1:
            # Second standardized momenta is the standard deviation
            std = raw_data.std()
            standardized_momenta.append(std)
        if maximum_momentum > 2:
            for i in range(2, maximum_momentum):
                # The rest of the standardized momenta are the centered raw data
                centered_momenta = ((raw_data - mean)**(i+1)).mean()
                centered_momenta = centered_momenta/(std**(i+1))*std
                standardized_momenta.append(centered_momenta)
        return torch.tensor(standardized_momenta, dtype=torch.float)


class SurfaceAreaDensityExtractor(BaseExtractor):
    """Extract surface area density and mean curvature."""

    def __init__(self, voxel_size: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.voxel_size = voxel_size

    @property
    def name(self) -> str:
        return "surface_area_density"

    @property
    def output_keys(self) -> List[str]:
        return ["surface_area_density", "mean_curvature"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.MIXED

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        from . import surface_area

        data_np = (1 - data[0].long()).numpy()
        sa = surface_area.surface_area_density(data_np, self.voxel_size)
        sa = torch.tensor(sa, dtype=torch.float)
        if sa.numel() == 0 or np.isnan(sa):
            sa = torch.tensor([0.0], dtype=torch.float)

        if len(data_np.shape) == 3:
            from . import curvature
            mean_curvature_integral = curvature.compute_mean_curvature_integral(data_np)
            mean_curvature_integral = mean_curvature_integral * self.voxel_size
            sa_int = sa * data_np.shape[0] * data_np.shape[1] * data_np.shape[2] * self.voxel_size**2
            if sa_int == 0.0 or np.array(mean_curvature_integral).size == 0:
                mean_curvature = torch.tensor([0.0], dtype=torch.float)
            else:
                mean_curvature = torch.tensor(mean_curvature_integral, dtype=torch.float) / sa_int
        else:
            mean_curvature = torch.tensor(np.nan, dtype=torch.float)

        return {"surface_area_density": sa, "mean_curvature": mean_curvature}


class EulerNumberDensityExtractor(BaseExtractor):
    """Extract Euler number density."""

    def __init__(self, voxel_size: float = 1.0, mode: str = "voxel", **kwargs):
        super().__init__(**kwargs)
        self.voxel_size = voxel_size
        self.mode = mode

    @property
    def name(self) -> str:
        return "euler_number_density"

    @property
    def output_keys(self) -> List[str]:
        return ["euler_number_density"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.THREE_D

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        from . import filters

        voxel = data[0].numpy().astype(bool)
        processed_voxel = filters.fill_blind_pores(voxel, conn=6)
        processed_voxel = ~filters.fill_blind_pores(~processed_voxel, conn=6)

        if self.mode == "voxel":
            euler_number = skimage.measure.euler_number(1 - processed_voxel, connectivity=3)
        elif self.mode == "mesh":
            from . import curvature
            m3 = curvature.compute_mean_curvature_integral(1 - processed_voxel, which="gaussian")
            euler_number = m3 / (4 * np.pi)

        voxel_volume = np.prod(voxel.shape) * self.voxel_size**3
        euler_number_density = euler_number / voxel_volume
        euler_number_density = torch.tensor([euler_number_density], dtype=torch.float)

        if euler_number_density.numel() == 0 or np.isnan(euler_number_density):
            euler_number_density = torch.tensor([0.0], dtype=torch.float)

        return {"euler_number_density": euler_number_density}


class PermeabilityExtractor(BaseExtractor):
    """Extract permeability using pore network modeling."""

    def __init__(
        self,
        type_pnm: int = 1,
        voxel_length: float = 2.25e-6,
        calculate_pc_curve: bool = False,
        disable_parallelization: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.voxel_length = voxel_length
        self.calculate_pc_curve = calculate_pc_curve
        self.type_pnm = type_pnm
        self.disable_parallelization = disable_parallelization

    @property
    def name(self) -> str:
        return "permeability_from_pnm"

    @property
    def output_keys(self) -> List[str]:
        return ["permeability"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.THREE_D

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        from . import permeability_from_pnm
        try:
            perm = permeability_from_pnm.calculate_permeability_from_pnm(
                data,
                self.voxel_length,
                self.calculate_pc_curve,
                type_pnm=self.type_pnm,
                disable_parallelization=self.disable_parallelization
            )
            out = {"permeability": torch.tensor(perm['permeabilities'], dtype=torch.float)}
            if self.calculate_pc_curve:
                out['pc_curve'] = {
                    'pc': torch.tensor(perm['pc_curve']['pc'], dtype=torch.float),
                    'snwp': torch.tensor(perm['pc_curve']['snwp'], dtype=torch.float)}
            return out
        except Exception:
            # Keep evaluation running when PNM fails for a sample.
            # The evaluator will report which sample got NaN permeability.
            out = {"permeability": torch.full((3,), torch.nan, dtype=torch.float)}
            if self.calculate_pc_curve:
                out['pc_curve'] = {
                    'pc': torch.tensor([], dtype=torch.float),
                    'snwp': torch.tensor([], dtype=torch.float),
                }
            return out


class SliceExtractor(BaseExtractor):
    """Extract random slices from 3D volumes."""

    def __init__(self, axis: Union[str, None] = None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis  # 'x', 'y', 'z', or None (random)

    @property
    def name(self) -> str:
        base_name = "slice_from_voxel"
        if self.axis:
            return f"{self.axis}{base_name}"
        return base_name

    @property
    def output_keys(self) -> List[str]:
        return ["slice"]

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.THREE_D

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.axis == 'x':
            slice_ind = np.random.randint(0, data.shape[2])
            slice_data = data[:, slice_ind, :, :]  # [1, H, W, Z] -> [1, H, W]
        elif self.axis == 'y':
            slice_ind = np.random.randint(0, data.shape[1])
            slice_data = data[:, :, slice_ind, :]  # [1, H, W, Z] -> [1, H, Z]
        elif self.axis == 'z':
            slice_ind = np.random.randint(0, data.shape[0])
            slice_data = data[:, :, :, slice_ind]  # [1, H, W, Z] -> [1, H, W]
        else:  # Random axis
            which_slice = np.random.randint(0, 3)
            if which_slice == 0:
                return SliceExtractor(axis='x').extract(data)
            elif which_slice == 1:
                return SliceExtractor(axis='y').extract(data)
            else:
                return SliceExtractor(axis='z').extract(data)

        return {"slice": slice_data}


class RandomSliceExtractor(BaseExtractor):
    """Base class for extractors that work on random slices from 3D volumes."""

    def __init__(self, base_extractor: BaseExtractor, **kwargs):
        super().__init__(**kwargs)
        self.base_extractor = base_extractor

    @property
    def name(self) -> str:
        return f"{self.base_extractor.name}_from_voxel_slice"

    @property
    def output_keys(self) -> List[str]:
        return self.base_extractor.output_keys

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.THREE_D

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract random slice
        ind = np.random.randint(0, data.shape[0])
        slice_data = data[ind, :, :]
        # Apply base extractor to slice
        return self.base_extractor.extract(slice_data)


class FeatureExtractorRegistry:
    """Registry for managing feature extractors."""

    def __init__(self):
        self._extractors = {}
        self._register_default_extractors()

    def register(self, extractor_class: type, name: str = None):
        """Register an extractor class."""
        if name is None:
            # Create instance to get name
            instance = extractor_class()
            name = instance.name
        self._extractors[name] = extractor_class

    def get_extractor(self, name: str, **kwargs) -> BaseExtractor:
        """Get an extractor instance by name."""
        if name not in self._extractors:
            raise ValueError(f"Extractor '{name}' not found. Available: {list(self._extractors.keys())}")
        return self._extractors[name](**kwargs)

    def list_extractors(self) -> List[str]:
        """List all available extractor names."""
        return list(self._extractors.keys())

    def get_extractor_info(self, name: str) -> Dict[str, Any]:
        """Get information about an extractor."""
        if name not in self._extractors:
            raise ValueError(f"Extractor '{name}' not found")

        instance = self._extractors[name]()
        return {
            "name": instance.name,
            "output_keys": instance.output_keys,
            "supported_dimensions": instance.supported_dimensions.value
        }

    def _register_default_extractors(self):
        """Register all default extractors."""
        # Basic extractors
        self.register(PorosityExtractor)
        self.register(EffectivePorosityExtractor)
        self.register(TwoPointCorrelationExtractor)
        self.register(PorosimetryExtractor)
        self.register(SurfaceAreaDensityExtractor)
        self.register(EulerNumberDensityExtractor)
        self.register(PermeabilityExtractor)
        self.register(SlicePorosityExtractor)
        self.register(SubvolumePorosityExtractor)

        # Slice extractors
        self.register(SliceExtractor, "slice_from_voxel")
        self.register(lambda **kwargs: SliceExtractor(axis='x', **kwargs), "xslice_from_voxel")
        self.register(lambda **kwargs: SliceExtractor(axis='y', **kwargs), "yslice_from_voxel")
        self.register(lambda **kwargs: SliceExtractor(axis='z', **kwargs), "zslice_from_voxel")

        # Variant extractors for different input types
        self._register_variants()

    def _register_variants(self):
        """Register variants of extractors for different input dimensions and sources."""
        # Create slice variants
        base_extractors = [
            ("two_point_correlation", TwoPointCorrelationExtractor),
            ("porosimetry", PorosimetryExtractor),
            ("surface_area_density", SurfaceAreaDensityExtractor),
        ]

        for base_name, base_class in base_extractors:
            # From slice variants
            self.register(base_class, f"{base_name}_from_slice")
            # From voxel variants (same as base)
            self.register(base_class, f"{base_name}_from_voxel")
            # From voxel slice variants
            self.register(
                lambda base_cls=base_class, **kwargs: RandomSliceExtractor(base_cls(**kwargs)),
                f"{base_name}_from_voxel_slice"
            )


class FeatureExtractorManager:
    """High-level interface for feature extraction."""

    def __init__(self):
        self.registry = FeatureExtractorRegistry()

    def create_extractor(self, name: str, input_type: InputType = InputType.BINARY, **kwargs) -> BaseExtractor:
        """Create a feature extractor with optional input preprocessing."""
        base_extractor = self.registry.get_extractor(name, **kwargs)

        if input_type == InputType.EDT:
            return EDTExtractor(base_extractor)

        return base_extractor

    def create_composite_extractor(self,
                                 extractor_names: List[str],
                                 extractor_kwargs: Dict[str, KwargsType] = None) -> 'CompositeExtractor':
        """Create a composite extractor from multiple extractors."""
        if extractor_kwargs is None:
            extractor_kwargs = {}

        extractors = []
        for name in extractor_names:
            kwargs = extractor_kwargs.get(name, {})
            extractors.append(self.create_extractor(name, **kwargs))

        return CompositeExtractor(extractors)

    def get_preset_extractors(self, preset: str) -> List[str]:
        """Get predefined sets of extractors."""
        presets = {
            "3d": [
                "porosimetry_from_voxel",
                "two_point_correlation_from_voxel",
                "permeability_from_pnm",
                "porosity",
                "effective_porosity",
                "surface_area_density_from_voxel",
                "euler_number_density"
            ],
            "2d": [
                "porosimetry_from_slice",
                "two_point_correlation_from_slice",
                "porosity",
                "effective_porosity",
                "surface_area_density_from_slice"
            ]
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")

        return presets[preset]

    def list_extractors(self) -> List[str]:
        """List all available extractors."""
        return self.registry.list_extractors()

    def get_extractor_info(self, name: str) -> Dict[str, Any]:
        """Get information about an extractor."""
        return self.registry.get_extractor_info(name)


class EDTExtractor(BaseExtractor):
    """Wrapper for extractors that need EDT input preprocessing."""

    def __init__(self, base_extractor: BaseExtractor):
        super().__init__()
        self.base_extractor = base_extractor

    @property
    def name(self) -> str:
        return self.base_extractor.name

    @property
    def output_keys(self) -> List[str]:
        return self.base_extractor.output_keys

    @property
    def supported_dimensions(self) -> DataDimension:
        return self.base_extractor.supported_dimensions

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Binarize EDT data
        binarized_data = data > 0
        return self.base_extractor.extract(binarized_data)


class CompositeExtractor(BaseExtractor):
    """Composite extractor that combines multiple extractors."""

    def __init__(self, extractors: List[BaseExtractor]):
        super().__init__()
        self.extractors = extractors

    @property
    def name(self) -> str:
        return "composite"

    @property
    def output_keys(self) -> List[str]:
        keys = []
        for extractor in self.extractors:
            keys.extend(extractor.output_keys)
        return keys

    @property
    def supported_dimensions(self) -> DataDimension:
        return DataDimension.MIXED

    def extract(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = {}
        for extractor in self.extractors:
            result.update(extractor.extract(data))
        return result


# Global instances for backward compatibility
_manager = FeatureExtractorManager()


# Backward compatibility functions
def make_feature_extractor(extractor_name: str,
                          input_type: Literal['binary', 'edt'] = 'binary',
                          **kwargs):
    """Create a feature extractor (backward compatibility)."""
    input_type_enum = InputType.BINARY if input_type == 'binary' else InputType.EDT
    extractor = _manager.create_extractor(extractor_name, input_type_enum, **kwargs)
    return extractor


def make_composite_feature_extractor(extractor_names: List[str],
                                   extractor_kwargs: Dict[str, KwargsType] = None):
    """Create a composite feature extractor (backward compatibility)."""
    return _manager.create_composite_extractor(extractor_names, extractor_kwargs)


# Constants for backward compatibility
AVAILABLE_EXTRACTORS = _manager.list_extractors()

EXTRACTORS_RETURN_KEYS_MAP = {
    name: _manager.get_extractor_info(name)["output_keys"]
    for name in AVAILABLE_EXTRACTORS
}


# Legacy extract functions for backward compatibility
def extract_composite(extractors: List[Callable]):
    """Legacy composite extractor function."""
    def composite_feature_extractor(x):
        data = {}
        for extractor in extractors:
            data.update(extractor(x))
        return data
    return composite_feature_extractor


# Helper function for standardized momenta (used by PorosimetryExtractor)
def calculate_standardized_momenta(raw_data, maximum_momentum):
    """Calculate the standardized momenta (backward compatibility)."""
    extractor = PorosimetryExtractor()
    return extractor._calculate_standardized_momenta(raw_data, maximum_momentum)


# Export the main interface
__all__ = [
    'FeatureExtractorManager',
    'BaseExtractor',
    'InputType',
    'DataDimension',
    'make_feature_extractor',
    'make_composite_feature_extractor',
    'AVAILABLE_EXTRACTORS',
    'EXTRACTORS_RETURN_KEYS_MAP'
]
