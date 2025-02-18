import porespy
import openpnm
import numpy as np

from ._snow2 import snow2


def calculate_permeability_from_pnm(volume, voxel_length):
    # Convert volume to binary (assuming 0 is pore space)
    binary_volume = (1 - volume[0].long().numpy())

    # Get volume dimensions
    volume_length = voxel_length * binary_volume.shape[0]

    # Generate network using SNOW algorithm
    partitioning = snow2(
        binary_volume,
        voxel_size=voxel_length,
    )

    # Convert to OpenPNM network
    pn = openpnm.io.network_from_porespy(partitioning.network)

    # Set up network properties
    pn['pore.diameter'] = pn['pore.equivalent_diameter']
    pn['throat.diameter'] = pn['throat.inscribed_diameter']
    pn['throat.spacing'] = pn['throat.total_length']

    pn_model = (
        openpnm.models.geometry.hydraulic_size_factors.pyramids_and_cuboids
    )
    pn.add_model(propname='throat.hydraulic_size_factors',
                 model=pn_model)
    pn.add_model(propname='throat.diffusive_size_factors',
                 model=pn_model)
    pn.regenerate_models()

    # Check and fix network health
    h = openpnm.utils.check_network_health(pn)
    openpnm.topotools.trim(network=pn, pores=h['disconnected_pores'])

    # Set up phase
    gas = openpnm.phase.Phase(network=pn)
    gas['pore.diffusivity'] = 1.0
    gas['pore.viscosity'] = 1.0
    gas.add_model_collection(openpnm.models.collections.physics.basic)
    gas.regenerate_models()

    # Calculate permeability in all three directions
    permeabilities = []
    for direction in ['x', 'y', 'z']:
        sf = openpnm.algorithms.StokesFlow(network=pn, phase=gas)
        sf.set_value_BC(pores=pn.pores(f'{direction}min'), values=1.0)
        sf.set_value_BC(pores=pn.pores(f'{direction}max'), values=0.0)
        sf.run()
        dP = 1.0
        L = volume_length
        A = volume_length**2
        K = sf.rate(pores=pn.pores(f'{direction}min'))*(L/A)/dP*1e12  # Darcy
        permeabilities.append(K[0])

    return np.array(permeabilities)
