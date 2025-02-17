import torch


class ForestFire3D(object):
    """
    Attributes
    ----------
    volume: Array[int, [W, H, D]]
        The 3D volume to be burned, where 1 indicates a
        flamable region and 0 is non-flamable
    W: int
        The width of the volume
    H: int
        The height of the volume
    D: int
        The depth of the volume
    burned: Set[Tuple[int, int, int]]
        The set of points that have been burned
    burned_frontiers: Set[Tuple[Tuple[int, int, int],
                          Tuple[int, int, int]]]
        The set of boundaries between burned and non-flammable.
        Only tracked for face neighbors mode.
    burned_boundaries: Various sets for each face of the volume
        The sets of boundaries between burned and the volume boundaries.
        Only tracked for face neighbors mode.
    """
    def __init__(self, volume, mark_burns=False, connectivity='face'):
        """
        Parameters
        ----------
        volume: Array[int, [W, H, D]]
            The volume to be burned, where 1 indicates a
            flamable region and 0 is non-flamable
        mark_burns: bool
            Whether to mark burned points with a burned flag
        connectivity: str
            Type of neighborhood connectivity: 'face' (6-connected),
            'edge' (18-connected), or 'corner' (26-connected)
        """
        self.volume = volume
        self.W = volume.shape[0]
        self.H = volume.shape[1]
        self.D = volume.shape[2]
        self.burned = set()
        self.connectivity = connectivity.lower()
        if self.connectivity not in ['face', 'edge', 'corner']:
            raise ValueError("connectivity must be 'face', 'edge', or 'corner'")

        # Only initialize boundary tracking for face neighbors
        if self.connectivity == 'face':
            self.burned_frontiers = set()
            self.burned_bottom = set()
            self.burned_top = set()
            self.burned_left = set()
            self.burned_right = set()
            self.burned_front = set()
            self.burned_back = set()

        self.mark_burns = mark_burns
        self.burn_mark = 0

    def increment_vectors(self):
        """
        Returns the increment vectors based on connectivity type
        """
        # Face neighbors (6-connected)
        face_neighbors = [
            [-1, 0, 0], [1, 0, 0],  # x direction
            [0, -1, 0], [0, 1, 0],  # y direction
            [0, 0, -1], [0, 0, 1]   # z direction
        ]

        # Edge neighbors (additional 12 vectors for 18-connected)
        edge_neighbors = [
            [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],  # xy plane
            [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],  # xz plane
            [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1]   # yz plane
        ]

        # Corner neighbors (additional 8 vectors for 26-connected)
        corner_neighbors = [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ]

        if self.connectivity == 'face':
            return face_neighbors
        elif self.connectivity == 'edge':
            return face_neighbors + edge_neighbors
        else:  # corner
            return face_neighbors + edge_neighbors + corner_neighbors

    def burn(self, start):  # noqa: C901
        """
        Burns the volume starting at the given point

        Parameters
        ----------
        start: Tuple[int, int, int]
            The starting point for the fire
        """
        is_flammable = self.volume[start[0], start[1], start[2]] == 1
        is_already_burned = start in self.burned
        if not is_flammable and not is_already_burned:
            return
        self.burn_mark -= 1
        stack = [start]
        while len(stack) > 0:
            current = stack.pop()
            self.burned.add(current)
            if self.mark_burns:
                self.volume[current[0], current[1], current[2]] = self.burn_mark
            x, y, z = current

            for dx, dy, dz in self.increment_vectors():
                next_x, next_y, next_z = x + dx, y + dy, z + dz

                # Check volume boundaries
                if (next_x < 0 or next_x >= self.W or
                        next_y < 0 or next_y >= self.H or
                        next_z < 0 or next_z >= self.D):
                    # For face neighbors, track boundaries
                    if self.connectivity == 'face':
                        (incx, incy, incz), (tx, ty, tz) = self.increment_map(dx, dy, dz)
                        bx, by, bz = x + incx, y + incy, z + incz
                        boundary_marking = ((bx, by, bz), (tx, ty, tz))

                        if next_x < 0:
                            self.burned_bottom.add(boundary_marking)
                        elif next_x >= self.W:
                            self.burned_top.add(boundary_marking)
                        elif next_y < 0:
                            self.burned_left.add(boundary_marking)
                        elif next_y >= self.H:
                            self.burned_right.add(boundary_marking)
                        elif next_z < 0:
                            self.burned_front.add(boundary_marking)
                        elif next_z >= self.D:
                            self.burned_back.add(boundary_marking)
                    continue

                # Check if next point is burned or non-flammable
                next = (next_x, next_y, next_z)
                is_not_flammable = self.volume[next[0], next[1], next[2]] == 0
                can_be_burned = self.volume[next[0], next[1], next[2]] == 1

                if is_not_flammable:
                    if self.connectivity == 'face':
                        (incx, incy, incz), (tx, ty, tz) = self.increment_map(dx, dy, dz)
                        bx, by, bz = x + incx, y + incy, z + incz
                        self.burned_frontiers.add(((bx, by, bz), (tx, ty, tz)))
                elif next in self.burned or not can_be_burned:
                    continue
                else:  # Add next point to stack
                    stack.append(next)

    def increment_map(self, dx, dy, dz):
        """
        Maps direction vectors to boundary coordinates and tangent vectors.
        Only used for face neighbors mode.
        """
        increment_map = {
            # x direction
            (1, 0, 0): ((1, 0, 0), (0, 1, 0)),
            (-1, 0, 0): ((0, 0, 0), (0, 1, 0)),
            # y direction
            (0, 1, 0): ((1, 1, 0), (-1, 0, 0)),
            (0, -1, 0): ((0, 0, 0), (1, 0, 0)),
            # z direction
            (0, 0, 1): ((1, 0, 1), (0, 1, 0)),
            (0, 0, -1): ((0, 0, 0), (1, 0, 0))
        }
        return increment_map[(dx, dy, dz)]

    # Rest of the methods remain unchanged
    def burn_points(self, points):
        """
        Burns the volume starting at the given points

        Parameters
        ----------
        points: List[Tuple[int, int, int]]
            The starting points for the fire
        """
        for point in points:
            self.burn(point)

    def burn_face(self, face):
        """
        Burns the volume starting at the specified face.
        Note: This method is most meaningful with face neighbors.

        Parameters
        ----------
        face: str
            One of 'bottom', 'top', 'left', 'right', 'front', 'back'
        """
        points = []
        if face == 'bottom':
            points = [(0, y, z) for y in range(self.H) for z in range(self.D)]
        elif face == 'top':
            points = [(self.W-1, y, z) for y in range(self.H) for z in range(self.D)]
        elif face == 'left':
            points = [(x, 0, z) for x in range(self.W) for z in range(self.D)]
        elif face == 'right':
            points = [(x, self.H-1, z) for x in range(self.W) for z in range(self.D)]
        elif face == 'front':
            points = [(x, y, 0) for x in range(self.W) for y in range(self.H)]
        elif face == 'back':
            points = [(x, y, self.D-1) for x in range(self.W) for y in range(self.H)]
        self.burn_points(points)

    def burn_all_faces(self):
        """
        Burns all faces of the volume.
        Note: This method is most meaningful with face neighbors.
        """
        self.burn_face('bottom')
        self.burn_face('top')
        self.burn_face('left')
        self.burn_face('right')
        self.burn_face('front')
        self.burn_face('back')

    def sample_from_bulk(self, nsamples):
        places = torch.tensor(list(self.burned),
                              dtype=torch.float)
        whichbox = torch.randint(0, places.shape[0], [nsamples])
        bulksamples = places[whichbox] + torch.rand([nsamples, 3])
        return bulksamples

    def sample_from_boundary(self, nsamples, which='frontier'):
        """
        Sample from boundaries. Only available in face neighbors mode.
        """
        if self.connectivity != 'face':
            raise ValueError("Boundary sampling only available in face neighbors mode")

        boundary_set = self.name_to_boundary_set(which)
        frontier_places, frontier_tangents = zip(*list(boundary_set))
        frontier_places = torch.tensor(frontier_places, dtype=torch.float)
        frontier_tangents = torch.tensor(frontier_tangents, dtype=torch.float)
        whichfrontier = torch.randint(0, frontier_places.shape[0], [nsamples])
        basesamples = torch.rand(nsamples, 1)
        frontiersamples = (frontier_places[whichfrontier] +
                           frontier_tangents[whichfrontier]*basesamples)
        return frontiersamples

    def name_to_boundary_set(self, which):
        """
        Get boundary set by name. Only available in face neighbors mode.
        """
        if self.connectivity != 'face':
            raise ValueError("Boundary sets only available in face neighbors mode")

        boundary_sets = {
            "frontier": self.burned_frontiers,
            "bottom": self.burned_bottom,
            "top": self.burned_top,
            "left": self.burned_left,
            "right": self.burned_right,
            "front": self.burned_front,
            "back": self.burned_back
        }
        return boundary_sets[which]