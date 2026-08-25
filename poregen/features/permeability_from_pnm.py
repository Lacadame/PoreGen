import openpnm
import numpy as np
import warnings

from .snow2 import snow2


def extract_pnm(volume, voxel_length, disable_parallelization: bool = False):
    binary_volume = (1 - volume[0].long().numpy())
    parallelization = None if disable_parallelization else {}
    partitioning = snow2(
        binary_volume,
        voxel_size=voxel_length,
        parallelization=parallelization,
    )
    pn = openpnm.io.network_from_porespy(partitioning.network)
    return pn


def calculate_permeability_from_pnm(volume, voxel_length, calculate_pc_curve=False,
                                    type_pnm=1, disable_parallelization: bool = False):
    # Convert volume to binary (assuming 0 is pore space)
    binary_volume = (1 - volume[0].long().numpy())

    # Get volume dimensions
    volume_length = voxel_length * binary_volume.shape[0]

    # Generate network using SNOW algorithm
    parallelization = None if disable_parallelization else {}
    partitioning = snow2(
        binary_volume,
        voxel_size=voxel_length,
        parallelization=parallelization,
    )

    # Convert to OpenPNM network
    if type_pnm == 1:
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

    elif type_pnm == 2:
        pn = openpnm.io.network_from_porespy(partitioning.network)
        pn['pore.diameter'] = pn['pore.equivalent_diameter']
        pn['throat.diameter'] = pn['throat.inscribed_diameter']
        pn['throat.spacing'] = pn['throat.total_length']
        pn['throat.radius'] = pn['throat.diameter'] / 2

        pn['throat.hydraulic_conductance'] = np.pi * (pn['throat.radius']**4) / (8 * pn['throat.spacing'])
    # Check and fix network health
    h = openpnm.utils.check_network_health(pn)
    openpnm.topotools.trim(network=pn, pores=h['disconnected_pores'])

    # Set up phase
    gas = openpnm.phase.Phase(network=pn)

    if type_pnm == 1:
        gas['pore.diffusivity'] = 1.0
        gas['pore.viscosity'] = 1.0
        gas.add_model_collection(openpnm.models.collections.physics.basic)
        gas.regenerate_models()

    # Calculate permeability in all three directions
    permeabilities = []
    for direction in ['x', 'y', 'z']:
        sf = openpnm.algorithms.StokesFlow(network=pn, phase=gas)
        inlet_pores = pn.pores(f'{direction}min')
        outlet_pores = pn.pores(f'{direction}max')
        if len(inlet_pores) == 0 or len(outlet_pores) == 0:
            permeabilities.append(np.nan)
            continue
        sf.set_value_BC(pores=inlet_pores, values=1.0)
        sf.set_value_BC(pores=outlet_pores, values=0.0)
        try:
            sf.run()
        except Exception as err:
            warnings.warn(
                f"Permeability solve failed in direction '{direction}': {err}. "
                "Returning NaN for this direction.",
                RuntimeWarning,
            )
            permeabilities.append(np.nan)
            continue

        dP = 1.0
        L = volume_length
        A = volume_length**2
        K = sf.rate(pores=inlet_pores)*(L/A)/dP*1e12  # Darcy
        permeabilities.append(K[0])

    out = {'permeabilities': np.array(permeabilities)}
    if calculate_pc_curve:
        pn['throat.volume'] = pn['throat.cross_sectional_area'] * 0
        hg = openpnm.phase.Mercury(network=pn)
        f = openpnm.models.physics.capillary_pressure.washburn
        hg.add_model(propname='throat.entry_pressure',
                     model=f,
                     surface_tension='throat.surface_tension',
                     contact_angle='throat.contact_angle',
                     diameter='throat.diameter',)
        mip = openpnm.algorithms.Drainage(network=pn, phase=hg)

        inlets = ['xmin', 'xmax', 'ymin', 'ymax']
        if binary_volume.ndim == 3:
            inlets = inlets + ['zmin', 'zmax']
        mip.set_inlet_BC(pores=pn.pores(inlets))  # mercury invades from all sides
        mip.run()

        data = mip.pc_curve()
        out['pc_curve'] = {'pc': data.pc, 'snwp': data.snwp}

    return out
