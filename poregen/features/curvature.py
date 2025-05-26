import pyvista
import numpy as np


def compute_mean_curvature_integral(subsample):
    """
    Compute the mean curvature integral for a binary volume.

    Parameters
    ----------
    subsample : ndarray
        Binary volume where the foreground (1's) represents the structure

    Returns
    -------
    W2 : float
        The mean curvature integral (Minkowski functional W2)
    """
    # Create isosurface
    surf = pyvista.wrap(subsample).contour()

    # Compute mean curvature at each point
    h = surf.curvature('mean')              # pointwise H

    # Compute area of each cell (triangle)
    surf = surf.compute_cell_sizes(area=True, length=False, volume=False)
    cell_area = surf.cell_data['Area']          # triangle area array

    # Distribute cell area to vertices (1/3 to each vertex of triangle)
    vertex_area = np.zeros(surf.n_points)
    faces = surf.faces.reshape(-1, 4)[:, 1:]     # triangle vertex ids
    vertex_area[faces] += (cell_area / 3)[:, None]

    # Compute integral of mean curvature (with factor 1/2 for Minkowski convention)
    W2 = 0.5 * np.dot(h, vertex_area)            # ∫ H dA

    return W2