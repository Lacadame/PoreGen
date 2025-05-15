from pathlib import Path
import warnings

import numpy as np
import torch

_LETTUCE_IMPORTED = False
try:
    import lettuce as lt
    _LETTUCE_IMPORTED = True
except ImportError:
    warnings.warn("Could not import lettuce. Some functionality may be limited.")


class PeriodicPressureBC(lt.Boundary):
    """According to Ehsan Evati: 'High performance simulation of fluid flow in porous media...'
    """

    def __init__(self, delta_rho: float, direction: list[int]):
        self.delta_rho = delta_rho
        self.direction = direction

        # Assert that direction has only one non-zero value
        non_zero_count = sum(1 for val in direction if val != 0)
        if non_zero_count != 1:
            raise ValueError("Direction must have exactly one non-zero value")

        # Assert that the non-zero value is either -1 or 1
        for val in direction:
            if val != 0 and val not in [-1, 1]:
                raise ValueError("Non-zero direction value must be either -1 or 1")

        # Find the index where direction is not 0
        self.ind = None
        for i, val in enumerate(self.direction):
            if val != 0:
                self.ind = i
                break

        # build indices of u and f that determine the side of the domain
        self.inlet_index = []
        self.outlet_index = []
        for i in direction:
            if i == 0:
                self.inlet_index.append(slice(None))
                self.outlet_index.append(slice(None))
            if i == 1:
                self.inlet_index.append(0)
                self.outlet_index.append(-1)
            if i == -1:
                self.inlet_index.append(-1)
                self.outlet_index.append(0)

    def __call__(self, flow: 'lt.Flow'):

        # Get stencil indexes whose e point along the direction
        stencil_e = flow.context.convert_to_ndarray(flow.stencil.e)
        # Find the indices where stencil_e points in the direction of flow
        inlet_indices = []
        for i in range(len(stencil_e)):
            if stencil_e[i, self.ind] == self.direction[self.ind]:
                inlet_indices.append(i)
        inlet_stencil_indexes = inlet_indices

        # Find the indices where stencil_e points opposite to the direction of flow
        outlet_indices = []
        for i in range(len(stencil_e)):
            if stencil_e[i, self.ind] == -self.direction[self.ind]:
                outlet_indices.append(i)

        w = flow.context.convert_to_tensor(flow.stencil.w)
        w_expanded = w
        for _ in range(len(flow.f.shape) - 2):
            w_expanded = w_expanded.unsqueeze(-1)

        f_neq = flow.f - flow.equilibrium(flow)
        f_eq_in = flow.equilibrium(flow, rho=flow.rho() + 2 * self.delta_rho)

        flow.f[[inlet_stencil_indexes] + self.inlet_index] = (
            f_neq[[inlet_stencil_indexes] + self.outlet_index] +
            f_eq_in[[inlet_stencil_indexes] + self.outlet_index]
        )

        return flow.f

    def make_no_collision_mask(self, shape: list[int], context: 'lt.Context') -> torch.Tensor | None:
        return None

    def make_no_streaming_mask(self, shape: list[int], context: 'lt.Context') -> torch.Tensor | None:
        pass

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int) -> 'lt.NativeBoundary':
        pass


class PorousMedium(lt.ExtFlow):
    """
    Flow class to simulate flow through a porous medium.
    """

    def __init__(self,
                 context,
                 resolution,
                 reynolds_number,
                 mach_number,
                 domain_length_x,
                 char_length=1,
                 char_velocity=1,
                 stencil=None,
                 equilibrium=None,
                 rho_drop=None,
                 direction=[1, 0, 0]):
        self.char_length_lu = 1.0
        self.char_length = char_length
        self.char_velocity = char_velocity
        self.rho_drop = rho_drop
        self.direction = direction
        self.resolution = self.make_resolution(resolution, stencil)
        self._mask = torch.zeros(self.resolution, dtype=torch.bool)
        lt.ExtFlow.__init__(self, context, resolution, reynolds_number,
                            mach_number, stencil, equilibrium)

    def make_units(self, reynolds_number, mach_number, resolution):
        return lt.UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_pu=self.char_length,
            characteristic_velocity_pu=self.char_velocity
        )

    def make_resolution(self, resolution, stencil=None):
        if isinstance(resolution, int):
            return [resolution] * (stencil.d or self.stencil.d)
        else:
            return resolution

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert ((isinstance(m, np.ndarray) or isinstance(m, torch.Tensor)) and
                all(m.shape[dim] == self.resolution[dim] for dim in range(
                    self.stencil.d)))
        self._mask = self.context.convert_to_tensor(m, dtype=torch.bool)

    def initial_pu(self):
        p = np.zeros_like(self.grid[0], dtype=float)[None, ...]
        u_char = 0.0 * self._unit_vector()
        u_char = lt.append_axes(u_char, self.stencil.d)
        u = ~self.mask * u_char
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(torch.arange(n)) for n in
                    self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        if self.rho_drop is None:
            return [lt.BounceBackBoundary(self.mask)]
        else:
            return [
                PeriodicPressureBC(self.rho_drop, self.direction),
                lt.BounceBackBoundary(self.mask)
            ]

    def _unit_vector(self, i=0):
        return torch.eye(self.stencil.d)[i]


def pad_with_zeros(subsample, buffer_size=20):
    """
    Create a padded version of subsample with buffer_size of zeros on all sides.

    Args:
        subsample: The 3D array to pad
        buffer_size: Number of zeros to pad on each side (default: 20)

    Returns:
        padded_subsample: The padded array
    """
    # Create a padded version of subsample with buffer_size of zeros on all sides
    padded_subsample = np.zeros((subsample.shape[0] + 2 * buffer_size,
                                 subsample.shape[1] + 2 * buffer_size,
                                 subsample.shape[2] + 2 * buffer_size),
                                dtype=subsample.dtype)

    # Insert the original data in the middle of the padded array
    padded_subsample[buffer_size:-buffer_size,
                     buffer_size:-buffer_size,
                     buffer_size:-buffer_size] = subsample

    # Now padded_subsample has buffer_size of zeros at the beginning and end of all axes
    return padded_subsample


def permeability_from_lbm(subsample,
                          grid_size_pu=2.25e-6,
                          padding_size=20,
                          mach_number=0.02,
                          reynolds_number=0.01,
                          acceleration=0.00001,
                          device="cuda:0",
                          it_max=10000,
                          it_check=100,
                          it_floating_avg=10,
                          avgs_per_check=20,
                          epsilon=1e-1,
                          savename=None,
                          mode='shanchen',
                          direction=[1, 0, 0]):
    """
    Calculate permeability from a 3D binary subsample using Lattice Boltzmann Method,
    in the direction of the x-axis.

    Args:
        subsample: 3D binary numpy array representing the porous medium (1=solid, 0=void)
        grid_size_pu: Grid size in physical units (default: 2.25e-6)
        padding_size: Number of cells to pad around the sample (default: 20)
        mach_number: Mach number for simulation (default: 0.02)
        reynolds_number: Reynolds number for simulation (default: 0.1)
        acceleration: Force acceleration parameter (default: 0.00001)
        device: Computation device (default: "cuda:0")
        it_max: Maximum number of iterations (default: 10000)
        it_check: Check convergence every it_check iterations (default: 100)
        it_floating_avg: Number of iterations to average over (default: 10)
        avgs_per_check: Number of iterations per check (default: 20)
        epsilon: Convergence criterion percentage (default: 1e-1)
        savepath: Path to save velocity and pressure fields (default: None)
        mode: 'shanchen' or 'pressuredrop'

    Returns:
        permeability: Calculated permeability value
        u_field: Velocity field after simulation (if savepath is not None)
        p_field: Pressure field after simulation (if savepath is not None)
    """
    if not _LETTUCE_IMPORTED:
        raise ImportError("Lettuce is not installed. Please install it using 'pip install lettuce'.")
    if direction != [1, 0, 0] and mode == 'shanchen':
        raise ValueError("Shan-Chen method is only available for x-direction.")

    # Ensure subsample is binary
    if not np.array_equal(subsample, subsample.astype(bool)):
        subsample = subsample.astype(bool)

    # Pad the subsample
    padded_subsample = pad_with_zeros(subsample, padding_size)

    # Setup context and stencil
    context = lt.Context(torch.device(device), use_native=False)
    if mode == 'pressuredrop':
        stencil = lt.D3Q19()
    else:
        stencil = lt.D3Q19()

    cs = 0.5773502691896258
    if mode == 'pressuredrop':
        rho_drop = padded_subsample.shape[0] / cs**2 * acceleration
    else:
        rho_drop = None

    # Create the flow
    flow = PorousMedium(context=context,
                        resolution=padded_subsample.shape,
                        reynolds_number=reynolds_number,
                        mach_number=mach_number,
                        domain_length_x=subsample.shape[0],
                        stencil=stencil,
                        rho_drop=rho_drop,
                        direction=direction)

    # Set the obstacles
    flow.mask = torch.tensor(padded_subsample, device=context.device, dtype=torch.bool)

    # Initialize collision with force
    if mode == 'shanchen':
        force = lt.ShanChen(flow=flow, tau=flow.units.relaxation_parameter_lu,
                            acceleration=acceleration)
        collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu, force=force)
    else:
        collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)

    # Create simulation
    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])

    # Run simulation until convergence
    u_avg = [np.Inf]
    for i in range(1, int(it_max//it_check)):
        u_avg_new = 0
        for j in range(it_floating_avg):
            simulation(avgs_per_check)
            u_avg_new += flow.u().mean()
        u_avg_new = u_avg_new/it_floating_avg
        u_avg.append(u_avg_new)
        rel_change = ((u_avg[-1]-u_avg[-2])/u_avg[-1]*100).abs()
        print(f'it {i*it_check} u_avg[-1] = {u_avg[-1]} the relative change in mean vel is {rel_change} %')
        if rel_change < epsilon or not u_avg[-1] == u_avg[-1]:
            break

    # Extract velocity field
    u_lu = flow.u()
    u_lu_masked = u_lu * (~flow.mask).float().to(u_lu.device)

    # Extract pressure field
    # p_lu = flow.p()
    # p_lu_masked = p_lu * (~flow.mask).float().to(p_lu.device)

    # Crop the fields to remove padding
    u_lu_masked_cropped = u_lu_masked[:,
                                      padding_size:-padding_size,
                                      padding_size:-padding_size,
                                      padding_size:-padding_size]

    # Calculate mean velocity in x-direction
    ind = 0 if direction == [1, 0, 0] else 1 if direction == [0, 1, 0] else 2
    u_x_mean = u_lu_masked_cropped[ind].mean()
    slice_indices = [slice(None), slice(None), slice(None)]
    slice_indices[ind] = -1
    u_surface = u_lu_masked_cropped[tuple(slice_indices)]
    u_surface_mean = u_surface[ind].mean()

    # Calculate permeability
    calculated_permeability_bulk = (u_x_mean / acceleration * flow.units.viscosity_lu).item()
    print(f"Calculated permeability bulk [LU]: {calculated_permeability_bulk}") 
    # Calculate surface permeability
    calculated_permeability_surface = (u_surface_mean / acceleration * flow.units.viscosity_lu).item()
    print(f"Calculated permeability surface [LU]: {calculated_permeability_surface}")
    if savename is not None:
        save_dir = Path(savename)
        save_dir.parent.mkdir(exist_ok=True, parents=True)

        # Convert tensors to numpy arrays
        u_field = context.convert_to_ndarray(u_lu_masked_cropped)

        # Ensure the savename has .npy extension
        if not savename.endswith('.npy'):
            savename = savename + '.npy'
        # Save the numpy arrays
        np.save(savename, u_field)

        print(f"Saved velocity (lattice units) to {save_dir}")

    return {'bulk': calculated_permeability_bulk, 'surface': calculated_permeability_surface}
