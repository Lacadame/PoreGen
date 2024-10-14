from typing import Callable, Any
import functools

import numpy as np
import porespy
import torch

from . import porosimetry
from . import permeability
from . import surface_area


KwargsType = dict[str, Any]


AVAILABLE_EXTRACTORS = [
    'two_point_correlation_from_voxel_slice',
    'two_point_correlation_from_voxel',
    'two_point_correlation_from_slice',
    'porosity',
    'porosimetry_from_voxel_slice',
    'porosimetry_from_voxel',
    'porosimetry_from_slice',
    'permeability_from_pnm',
    'surface_area_density_from_slice',
    'surface_area_density_from_voxel',
    'surface_area_densityfrom_voxel_slice'
]


# Extractors

def extract_two_point_correlation_base(data, bins: int = 32):
    data = data[0].float()
    tpc_data = porespy.metrics.two_point_correlation((1 - data).numpy(), bins=bins)
    dist = torch.tensor(tpc_data.distance, dtype=torch.float)
    prob = torch.tensor(tpc_data.probability_scaled, dtype=torch.float)
    prob = torch.nan_to_num(prob)
    return {'tpc_dist': dist, 'tpc_prob': prob}


def extract_two_point_correlation_from_voxel_slice(voxel, bins: int = 32):
    ind = np.random.randint(0, voxel.shape[0])
    slice = voxel[ind, :, :]
    return extract_two_point_correlation_base(slice, bins)


def extract_two_point_correlation_from_voxel(voxel, bins: int = 32):
    return extract_two_point_correlation_base(voxel, bins)


def extract_two_point_correlation_from_slice(slice, bins: int = 32):
    return extract_two_point_correlation_base(slice, bins)


def extract_porosimetry_base(data,
                             bins: int = 32,
                             log: bool = False,
                             maximum_momentum: int = 4):
    data = data[0].float()
    im = porosimetry.local_thickness((1 - data).numpy())
    psd_data = porespy.metrics.pore_size_distribution(im, bins=bins, log=log)
    bin_centers = torch.tensor(psd_data.bin_centers.copy(), dtype=torch.float)
    cdf = torch.tensor(psd_data.cdf.copy(), dtype=torch.float)
    pdf = torch.tensor(psd_data.pdf.copy(), dtype=torch.float)

    raw_data = im.flatten()[im.flatten() > 0]
    data_size = raw_data.shape[0]
    momenta = []
    for i in range(maximum_momentum):
        momenta.append((raw_data**(i+1)).sum(axis=0)/data_size)
    log_momenta = torch.tensor(np.log(np.array(momenta)), dtype=torch.float)
    root_momenta = momenta**np.array([1/(i+1) for i in range(maximum_momentum)])
    root_momenta = torch.tensor(root_momenta, dtype=torch.float)
    return {'psd_centers': bin_centers,
            'psd_cdf': cdf,
            'psd_pdf': pdf,
            'log_momenta': log_momenta,
            'root_momenta': root_momenta}


def extract_porosimetry_from_slice(slice,
                                   bins: int = 32,
                                   log: bool = False,
                                   maximum_momentum: int = 4):
    return extract_porosimetry_base(slice, bins, log, maximum_momentum)


def extract_porosimetry_from_voxel(voxel,
                                   bins: int = 32,
                                   log: bool = False,
                                   maximum_momentum: int = 4):
    return extract_porosimetry_base(voxel, bins, log, maximum_momentum)


def extract_porosimetry_from_voxel_slice(voxel,
                                         bins: int = 32,
                                         log: bool = False,
                                         maximum_momentum: int = 4):
    ind = np.random.randint(0, voxel.shape[0])
    slice = voxel[ind, :, :]
    return extract_porosimetry_base(slice, bins, log, maximum_momentum)


def extract_surface_area_density_base(data, voxel_size: float = 1.0):
    data = (1 - data[0].long()).numpy()
    sa = surface_area.surface_area_density(data, voxel_size)
    return {'surface_area_density': torch.tensor(sa, dtype=torch.float)}


def extract_surface_area_density_from_slice(slice, voxel_size: float = 1.0):
    return extract_surface_area_density_base(slice, voxel_size)


def extract_surface_area_density_from_voxel(voxel, voxel_size: float = 1.0):
    return extract_surface_area_density_base(voxel, voxel_size)


def extract_surface_area_density_from_voxel_slice(voxel, voxel_size: float = 1.0):
    ind = np.random.randint(0, voxel.shape[0])
    slice = voxel[ind, :, :]
    return extract_surface_area_density_base(slice, voxel_size)


def extract_porosity(slice):
    porosity = torch.tensor([(1 - slice.numpy().mean())], dtype=torch.float)
    return {'porosity': porosity}


def extract_permeability_from_pnm(voxel,
                                  voxel_length=2.25e-6):
    try:
        perm = permeability.calculate_permeability_from_pnm(voxel, voxel_length)
    except Exception:  # Could not calculate permeability
        perm = np.nan*np.ones(len(voxel.shape) - 1)
    return {'permeability': torch.tensor(perm, dtype=torch.float)}


# Composite extractor
def extract_composite(
        extractors: list[Callable]):
    def composite_feature_extractor(x):
        data = {}
        for extractor in extractors:
            data.update(extractor(x))
        return data

    return composite_feature_extractor


# Functions to make extractors

def make_feature_extractor(extractor_name: str,
                           **kwargs):
    # Get the extractor function
    if extractor_name not in AVAILABLE_EXTRACTORS:
        raise ValueError(f"Extractor {extractor_name} not available")
    # Extend the extractor name with extract_
    extractor_name_extended = f"extract_{extractor_name}"
    extractor_fn = globals()[extractor_name_extended]
    # Make the extractor function from partial
    extractor_fn = functools.partial(extractor_fn, **kwargs)
    return extractor_fn


def make_composite_feature_extractor(extractor_names: list[str],
                                     extractor_kwargs: dict[str, KwargsType] = {}):
    # Make the extractor functions
    extractors = []
    for name in extractor_names:
        args = extractor_kwargs.get(name, {})
        extractors.append(make_feature_extractor(name, **args))
    # Make the composite extractor
    return extract_composite(extractors)
