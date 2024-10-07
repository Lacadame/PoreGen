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
]


# Extractors

def extract_two_point_correlation_from_voxel_slice(
        voxel,
        bins: int = 100):
    voxel = voxel[0]
    ind = np.random.randint(0, voxel.shape[0])
    slice = voxel[ind, :, :]
    data = porespy.metrics.two_point_correlation((1 - slice).numpy(), bins=bins)
    dist = torch.tensor(data.distance, dtype=torch.float)
    prob = torch.tensor(data.probability_scaled, dtype=torch.float)
    prob = torch.nan_to_num(prob)

    return {'tpc_dist': dist, 'tpc_prob': prob}


def extract_two_point_correlation_from_voxel(
        voxel,
        bins: int = 100):
    voxel = voxel[0]
    data = porespy.metrics.two_point_correlation((1 - voxel).numpy(), bins=bins)
    dist = torch.tensor(data.distance, dtype=torch.float)
    prob = torch.tensor(data.probability_scaled, dtype=torch.float)
    prob = torch.nan_to_num(prob)

    return {'tpc_dist': dist, 'tpc_prob': prob}


def extract_two_point_correlation_from_slice(
        slice,
        bins: int = 100):
    slice = slice[0]
    data = porespy.metrics.two_point_correlation((1 - slice).numpy(), bins=bins)
    dist = torch.tensor(data.distance, dtype=torch.float)
    prob = torch.tensor(data.probability_scaled, dtype=torch.float)
    prob = torch.nan_to_num(prob)

    return {'tpc_dist': dist, 'tpc_prob': prob}


def extract_porosimetry_from_slice(
        slice,
        bins: int = 100,
        log: bool = False):
    slice = slice[0]
    im = porosimetry.local_thickness((1 - slice).numpy())
    data = porespy.metrics.pore_size_distribution(
        im,
        bins=bins,
        log=log)
    bin_centers = torch.tensor(data.bin_centers.copy(), dtype=torch.float)
    cdf = torch.tensor(data.cdf.copy(), dtype=torch.float)
    pdf = torch.tensor(data.pdf.copy(), dtype=torch.float)
    return {'psd_centers': bin_centers, 'psd_cdf': cdf, 'psd_pdf': pdf}


def extract_porosimetry_from_voxel(
        voxel,
        bins: int = 100,
        log: bool = False):
    voxel = voxel[0]
    im = porosimetry.local_thickness((1 - voxel).numpy())
    data = porespy.metrics.pore_size_distribution(
        im,
        bins=bins,
        log=log)
    bin_centers = torch.tensor(data.bin_centers.copy(), dtype=torch.float)
    cdf = torch.tensor(data.cdf.copy(), dtype=torch.float)
    pdf = torch.tensor(data.pdf.copy(), dtype=torch.float)
    return {'psd_centers': bin_centers, 'psd_cdf': cdf, 'psd_pdf': pdf}


def extract_porosimetry_from_voxel_slice(
        voxel,
        bins: int = 10,
        log: bool = False):
    ind = np.random.randint(0, voxel.shape[0])
    slice = voxel[ind, :, :]
    im = porosimetry.local_thickness((1 - slice).numpy())
    data = porespy.metrics.pore_size_distribution(
        im,
        bins=bins,
        log=log)
    bin_centers = torch.tensor(data.bin_centers.copy(), dtype=torch.float)
    cdf = torch.tensor(data.pdf.copy(), dtype=torch.float)
    pdf = torch.tensor(data.pdf.copy(), dtype=torch.float)
    return {'psd_centers': bin_centers, 'psd_cdf': cdf, 'psd_pdf': pdf}


def extract_porosity(slice):
    porosity = torch.tensor([(1 - slice.numpy().mean())], dtype=torch.float)
    return {'porosity': porosity}


def extract_permeability_from_pnm(voxel,
                                  voxel_length=2.25e-6):
    perm = permeability.calculate_permeability_from_pnm(voxel, voxel_length)
    return {'permeability': perm}


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
