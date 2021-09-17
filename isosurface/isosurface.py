# Author: Kevin Juan

import MDAnalysis as mda
import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from numba import jit
from skimage import measure
from scipy.interpolate import griddata
import itertools as it
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay


class density_field:
    """
    Args:
        box_dim(np.ndarray): Array of [Lx, Ly, Lz, alpha, beta, gamma]
        n_grid(np.ndarray): Array of [Nx, Ny, Nz] where Ni is the number of grid points in the i-direction
    """

    def __init__(self, positions, box_dim, n_grid, sigma=3.1, n=2.5, ns_grid=True):
        # Atom positions
        self.positions = positions

        # Box dimensions
        self.box_dim = box_dim
        self.Lx, self.Ly, self.Lz = box_dim[:3]

        # Grid spacing
        self.n_grid = n_grid
        self.Nx, self.Ny, self.Nz = n_grid
        self.delta = box_dim[:3] / n_grid
        self.dx, self.dy, self.dz = box_dim[:3] / n_grid

        # Coarse graining and other parameters
        self.sigma = sigma
        self.n = n
        self.L = n * sigma
        self.discretize(ns_grid)

    def discretize(self, ns_grid=True):
        """
        Subroutine that prepares the simulation box to get instantaneous isosurface. MDAnalysis needs things in np.float32

        Args:
            ns_grid(bool): Determines whether or not to use cell list grid searching
        """
        # Generate the meshgrid
        self.xx, self.yy, self.zz = np.mgrid[0:self.Lx:complex(self.Nx), 0:self.Ly:complex(self.Ny), 0:self.Lz:complex(self.Nz)]

        self.grids = np.vstack((self.xx.flatten(), self.yy.flatten(), self.zz.flatten())).T.astype(np.float32)

        # Generate cell list
        if ns_grid:
            self.grid_search = mda.lib.nsgrid.FastNS(self.n * self.sigma, self.grids, self.box_dim, pbc=True)
        else:
            self.grid_search = None

    def get_grid(self):
        """
        Getter function for the field meshgrid and its parameters.

        Return:
            self.grids(np.ndarray): Simulation discretized into [Nx, Ny, Nz] points
            self.delta(np.ndarray): Vector containing dx, dy, and dz
            self.n_grid(np.ndarray): Vector containing Nx, Ny, Nz
        """

        return self.grids, self.delta, self.n_grid

    def density_field_grid_search(self, d=3):
        """
        Generates the density field by cutting off the coarse-graining Gaussian at n * sigma. Meshgrid points are
        in this cutoff are found using cell list approach. This was adapted from Yusheng's code, which uses KDTrees.

        Args:
            positions(np.ndarray): [N, 3] array of the atom positions
            d(float): Dimensionality for the field

        Return:
            field(np.ndarray): [Nx, Ny, Nz] array of the density field
        """
        # Generate an array of zeros to store the field
        field = np.zeros((self.Nx * self.Ny * self.Nz))

        # MDAnalysis grid search using cell list
        ns_grid_results = self.grid_search.search(self.positions.astype(np.float32))
        pairs = ns_grid_results.get_pairs()
        dr = ns_grid_results.get_pair_distances()

        # The pairs should be sorted
        sorted_idx = pairs[:, 0].argsort()
        pairs_sorted = pairs[sorted_idx]
        dr_sorted = dr[sorted_idx]

        # For each atom group the grid indices and distances that lie near it
        # Each grouping will be a list of length N_atoms
        nn_grid_idx = np.split(pairs_sorted[:, 1], np.unique(pairs_sorted[:, 0], return_index=True)[1][1:])
        nn_grid_dr = np.split(dr_sorted, np.unique(pairs_sorted[:, 0], return_index=True)[1][1:])

        for atom_idx, grid_idx in enumerate(nn_grid_idx):
            field[grid_idx] += coarse_grain(nn_grid_dr[atom_idx], self.sigma, d).astype('float32')

        field = field.reshape(self.Nx, self.Ny, self.Nz)

        return field

    def iso_3d_mcubes(self, field, c=0.015, gradient='descent'):
        """
        Generates the positions of triangles used for plotting the isosurface from marching cubes

        Args:
            field(np.ndarray): The density field in shape [Nx, Ny, Nz]
            c(float): Countour line value for the isosurface (default is 0.016 waters/Angstrom^3)
            gradient(str): 'descent' if values exterior of object are smaller, 'ascent' if values exterior of object are larger

        Return:
            verts(np.ndarray): Array of vertices with shape [V, 3]
            faces(np.ndarray): Array of triangle faces with shape [F, 3]
        """
        dx, dy, dz = self.box_dim[:3] / (self.n_grid - 1)

        verts, faces, normals, values = measure.marching_cubes(field, c, spacing=(dx, dy, dz), gradient_direction=gradient)

        return verts.astype('float32'), faces

    def iso_3d_grid(self, verts, box=False, direction=2, n_grid=None, type='linear'):
        """
        Generates an isosurface on a regularly spaced grid by interpolating from triangle vertices
        Corrects NaN values using nearest mode

        Args:
            verts(np.ndarray): Array of shape [V, 3]
            box(bool): Whether to use the entire box or the input surface
            plane(int): The direction normal to the surface
            n_grid(np.ndarray): Array of [Nx, Ny, Nz] to interpolate the grid (default is the same used for the field)
            type(str): Method used for interpolating the grid (linear and cubic are recommended)

        Return:
            surface(np.ndarray): Array of shape [N1, N2] that contains the points of the surface
            grid1(np.ndarray): Array of coordinates for dimension 1 of the grid
            grid2(np.ndarray): Array of coordinates for dimension 2 of the grid
        """
        if n_grid is None:
            n_grid = self.n_grid

        # Do the interpolation over the input surface
        # Else, do the interpolation over the entire box
        if box is False:
            x_max, x_min = np.max(verts[:, 0]), np.min(verts[:, 0])
            y_max, y_min = np.max(verts[:, 1]), np.min(verts[:, 1])
            z_max, z_min = np.max(verts[:, 2]), np.min(verts[:, 2])
        else:
            x_max, x_min = self.Lx, 0
            y_max, y_min = self.Ly, 0
            z_max, z_min = self.Lz, 0

        if direction == 2:
            points = verts[:, :2]
            values = verts[:, 2]
            grid_1, grid_2 = np.mgrid[x_min:x_max:complex(n_grid[0]), y_min:y_max:complex(n_grid[1])]
        elif direction == 0:
            points = verts[:, 1:]
            values = verts[:, 0]
            grid_1, grid_2 = np.mgrid[0:y_max:complex(n_grid[1]), z_min:z_max:complex(n_grid[2])]
        else:
            points = verts[:, 0::2]
            values = verts[:, 1]
            grid_1, grid_2 = np.mgrid[x_min:x_max:complex(n_grid[0]), z_min:z_max:complex(n_grid[2])]

        surface = griddata(points, values, (grid_1, grid_2), method=type)
        surface = surface.flatten()

        # Fill in any NaN by using nearest interpolation scheme
        cond = np.where(np.isnan(surface))[0]
        if cond.shape[0] != 0:
            surface_ = griddata(points, values, (grid_1, grid_2), method='nearest')
            surface[cond] = surface_.flatten()[cond]

        return np.c_[grid_1.flatten(), grid_2.flatten(), surface].astype('float32')

    def split_mcubes_surf(self, verts, faces, direction=2, defects=False, n_expected=2):
        """
        Separates the marching cubes isosurface data into distinct surfaces
        The results will always sort the surface in descending order relative to the direction normal to the surface (i.e. highest surface first)
        Uses the faces to determine number of connected graphs to retrieve unique surfaces
        Can optionally return the isosurfaces for defects
        Note: Indices in faces_ are relative to the input data since reindexing is non-trivial

        Args:
            verts(np.ndarray): Array of vertices with shape [V, 3]
            faces(np.ndarray): Array of triangle faces with shape [F, 3]
            n_surf(int): Number of expected surfaces
            direction(int): Axis normal to the plane that the surface lies in
            defects(bool): Whether to output the defects
            n_expected(int): Number of true surfaces anticipated

        Return:
            faces_surface(np.ndarray): Array of length n_surf containing the faces for the different surfaces
            verts_surface(np.ndarray): Array of length n_surf containing the vertices for the different surfaces
            faces_defect(np.ndarray): Array of length n_surf containing the faces for the different defects
            verts_defect(np.ndarray): Array of length n_surf containing the vertices for the different defects
        """
        faces_ = []
        verts_ = []

        edges = []
        for face in faces:
            edges.extend(list(it.combinations(face, 2)))
        g = nx.from_edgelist(edges)

        # Compute connected components and print results
        components = list(nx.algorithms.components.connected_components(g))

        # Need to determine if number of verts/faces are same
        # for both all surfaces to convert to object array or not
        n_verts = np.zeros(len(components))
        for i, component in enumerate(components):
            faces_list = np.array([face for face in faces if set(face) <= component])
            faces_.append(faces_list) # <= operator tests for subset relation
            verts_.append(verts[np.unique(faces_list)])
            n_verts[i] = len(verts[np.unique(faces_list)])

        # Sort the surfaces by size to find defects and the true surfaces
        if np.all(n_verts == n_verts[0]):
            verts_ = np.array(verts_)
        else:
            verts_ = np.array(verts_, dtype=object)

        cond_size = np.argsort([i.shape[0] for i in verts_])[::-1]
        verts_ = verts_[cond_size]

        faces_ = []
        n_faces = np.zeros(len(components))
        # Recompute the faces
        if direction == 2:
            for i, v in enumerate(verts_):
                faces_.append(Delaunay(v[:, :2]).simplices)
                n_faces[i] = len(faces_[i])
        elif direction == 1:
            for i, v in enumerate(verts_):
                faces_.append(Delaunay(v[:, ::2]).simplices)
                n_faces[i] = len(faces_[i])
        else:
            for i, v in enumerate(verts_):
                faces_.append(Delaunay(v[:, 1:]).simplices)
                n_faces[i] = len(faces_[i])

        if np.all(n_faces == n_faces[0]):
            faces_ = np.array(faces_)
        else:
            faces_ = np.array(faces_, dtype=object)

        # Discriminate between true surfaces and defects
        verts_surface = verts_[:n_expected]
        faces_surface = faces_[:n_expected]
        verts_defect = verts_[n_expected:]
        faces_defect = faces_[n_expected:]

        # Sort interface surfaces by height
        cond_height = np.argsort([np.average(i[:, direction]) for i in verts_surface])[::-1]
        verts_surface = verts_surface[cond_height]
        faces_surface = faces_surface[cond_height]

        return (verts_surface, faces_surface, verts_defect, faces_defect) if defects else (verts_surface, faces_surface)

    def area(self, verts, faces):
        """
        Returns the area of the isosurface

        Args:
            verts(np.ndarray): Array of vertices with shape [V, 3]
            faces(np.ndarray): Array of triangle faces with shape [F, 3]

        Return:
            area(float): Total surface area
        """
        area = measure.mesh_surface_area(verts, faces)

        return area

    def volume(self, prism, direction=2):
        """
        Returns the volume bounded by two surfaces

        Args:
            prism(np.ndarray): Array of triangles with shape [T1, 3] where one column contains the heights
            direction(int): Axis normal to the plan that the surfaces lie in

        Return:
            vol(float): Total volume bounded by the surfaces
        """
        # Vary calculation based on the axis of growth
        if direction == 2:
            # Calculate edges for base area
            BA = prism[:, :, :2][:, 1] - prism[:, :, :2][:, 0]
            CA = prism[:, :, :2][:, 2] - prism[:, :, :2][:, 0]
        elif direction == 1:
            # Calculate edges for base area
            BA = prism[:, :, ::2][:, 1] - prism[:, :, ::2][:, 0]
            CA = prism[:, :, ::2][:, 2] - prism[:, :, ::2][:, 0]
        else:
            # Calculate edges for base area
            BA = prism[:, :, 1:][:, 1] - prism[:, :, 1:][:, 0]
            CA = prism[:, :, 1:][:, 2] - prism[:, :, 1:][:, 0]

        # Base area
        area = 0.5 * np.cross(BA, CA)

        # Prism volume
        vol = np.sum(area * np.sum(prism[:, :, direction], axis=1) / 3)

        return vol

    def defect_vol(self, surface, verts_def, faces_def, direction=2):
        """
        Computes the volume of defects from marching cubes mesh

        Args:
            verts_def(np.ndarray): Array of vertices with shape [D, V, 3]
            faces_def(np.ndarray): Array of triangle faces with shape [D, F, 3]
            direction(int): Axis normal to the plane that the surface lies in

        Return:
            vol(np.ndarray): Array of volumes for each defect
            dist(np.ndarray): Array of defect distances to surface
        """
        if verts_def.shape[0] == 0:
            return np.zeros(1, dtype=np.float32), np.array([np.nan], dtype=np.float32)
        else:
            vol = np.zeros((verts_def.shape), dtype=np.float32)
            dist = np.zeros((verts_def.shape), dtype=np.float32)
            for i in range(verts_def.shape[0]):
                # Place an origin point within the defect
                origin_x = (np.max(verts_def[i][:, 0]) - np.min(verts_def[i][:, 0])) * 0.5 + np.min(verts_def[i][:, 0])
                origin_y = (np.max(verts_def[i][:, 1]) - np.min(verts_def[i][:, 1])) * 0.5 + np.min(verts_def[i][:, 1])
                origin_z = (np.max(verts_def[i][:, 2]) - np.min(verts_def[i][:, 2])) * 0.5 + np.min(verts_def[i][:, 2])
                origin = np.r_[origin_x, origin_y, origin_z]

                # Calculate the volume of the pyramids
                do = verts_def[i][faces_def[i]] - origin
                vol[i] = np.sum(np.abs((do[:, 0] * np.cross(do[:, 1], do[:, 2])).sum(-1))) / 6

                # Distance to surface
                p, t = moller_trumbore(surface, origin, direction) # Add direction parameter
                dist[i] = t

            return vol, dist

    def true_ice_nn(self, positions, surf1, surf2, box=None, direction=2):
        """
        Returns the number of atoms bound by a set of surfaces

        Args:
            positions(np.ndarray): Array of atom positions with shape [N, 3]
            surf1(np.ndarray): Array of vertices with shape [V1, 3]
            surf2(np.ndarray): Array of vertices with shape [V2, 3]
            box(np.ndarray): Vector of length 3 containing box dimensions
            direction(int): Axis normal to the plane that the surface lies in
            tol(float): Tolerance that determines what is "far" from the surface

        Return:
            n_true(int): Number of atoms bound by the surfaces
            n_true_idx(np.ndarray): Indices of true ice for smart indexing
            n_false_idx(np.ndarray): Indices of of false positive ice for smart indexing
        """
        if box is None:
            box = self.box_dim[:3]

        # Determine which is the upper and lower bounding surface
        if np.average(surf1[:, direction]) > np.average(surf2[:, direction]):
            upper = surf1
            lower = surf2
        else:
            upper = surf2
            lower = surf1

        # Select only atoms that are above the min for the top surface and below the max for the bottom surface
        cond = (positions[:, direction] > np.min(upper[:, direction])) | (positions[:, direction] < np.max(lower[:, direction]))
        true_idx = np.where(~cond)[0] # Indices for atoms that are well in the bulk
        idx_to_test = np.where(cond)[0] # Indices that we wish to check
        positions = positions[idx_to_test]

        # Compute diagonal length of the plane normal to growth direction
        # Also select the relevant atom coordinates
        if direction == 2:
            r = np.linalg.norm(box[:2])
            upper_surf_coords = upper[:, :2]
            lower_surf_coords = lower[:, :2]
            atom_coords = positions[:, :2]
        elif direction == 1:
            r = np.linalg.norm(box[::2])
            upper_surf_coords = upper[:, ::2]
            lower_surf_coords = lower[:, ::2]
            atom_coords = positions[:, ::2]
        else:
            r = np.linalg.norm(box[1:])
            upper_surf_coords = upper[:, 1:]
            lower_surf_coords = lower[:, 1:]
            atom_coords = positions[:, 1:]

        # Compute the 1-NN
        nn_upper = NearestNeighbors(n_neighbors=1, algorithm='auto', radius=r).fit(upper_surf_coords)
        dist_upper, idx_upper = nn_upper.kneighbors(atom_coords)
        nn_lower = NearestNeighbors(n_neighbors=1, algorithm='auto', radius=r).fit(lower_surf_coords)
        dist_lower, idx_lower = nn_lower.kneighbors(atom_coords)

        # Perform bound check
        upper_bound = upper[idx_upper.flatten(), direction] >= positions[:, direction]
        lower_bound = lower[idx_lower.flatten(), direction] <= positions[:, direction]
        bound_cond = np.all(np.c_[upper_bound, lower_bound], axis=1)
        true_idx = np.sort(np.append(true_idx, idx_to_test[np.where(bound_cond)[0]])).astype(np.int32)
        false_idx = np.sort(idx_to_test[np.where(~bound_cond)[0]]).astype(np.int32)

        # False positives near the surface
        upper_surf_dist = upper[idx_upper[np.where(~bound_cond)[0]].flatten(), direction] - positions[np.where(~bound_cond)[0], direction]
        lower_surf_dist = lower[idx_lower[np.where(~bound_cond)[0]].flatten(), direction] - positions[np.where(~bound_cond)[0], direction]
        false_dist = np.min(np.abs(np.c_[upper_surf_dist, lower_surf_dist]), axis=1).astype(np.float32)

        # Number of true ice
        n_true = true_idx.shape[0]

        return n_true, true_idx, false_idx, false_dist

    def true_ice_mt(self, positions, surf1, surf2, direction=2):
        """
        Moller-Trumbore algorithm for finding the intersection between a ray and triangle to calculate true lambda

        Args:
            positions(np.ndarray): Array of atom positions with shape [N, 3]
            surf1(np.ndarray): Array of triangles with shape [T1, 3]
            surf2(np.ndarray): Array of triangles shape [T2, 3]
            direction(int): Axis normal to the plane that the surface lies in

        Return:
            n_true(int): Number of atoms bound by the surfaces
            true_idx(np.ndarray): Indices of true ice for smart indexing
            false_idx(np.ndarray): Indices of false positive ice for smart indexing
            false_dist(np.ndarray): Distances of false positive ice to surface
        """
        # Determine which is the upper and lower bounding surface
        if np.average(surf1[:, :, direction]) > np.average(surf2[:, :, direction]):
            upper = surf1
            lower = surf2
        else:
            upper = surf2
            lower = surf1

        # Select only atoms that are above the min for the top surface and below the max for the bottom surface
        cond = (positions[:, direction] > np.min(upper[:, :, direction])) | (positions[:, direction] < np.max(lower[:, :, direction]))
        true_idx = np.where(~cond)[0] # Indices for atoms that are well in the bulk
        idx_to_test = np.where(cond)[0] # Indices that we wish to check
        positions = positions[idx_to_test]

        # Store false positive indices and distances
        false_idx = []
        false_dist = []

        for i, atom in enumerate(positions):

            # Find intersection point
            p_top, t_top = moller_trumbore(upper, atom, direction)
            p_bot, t_bot = moller_trumbore(lower, atom, direction)

            # Impose conditions to filter true ice from false positives
            if atom[direction] <= p_top[direction] and atom[direction] >= p_bot[direction]:
                true_idx = np.append(true_idx, idx_to_test[i]).astype(np.int32)
            else:
                false_idx.append(idx_to_test[i])
                false_dist.append(np.min(np.abs(np.c_[t_top, t_bot])))

        # Sort
        false_idx = np.array(false_idx, dtype=np.int32)
        false_dist = np.array(false_dist, dtype=np.float32)
        sorted = false_idx.argsort()
        false_idx = np.sort(false_idx)
        false_dist = false_dist[sorted]

        # Get true lambda
        n_true = true_idx.shape[0]

        return n_true, true_idx, false_idx, false_dist


@jit(nopython=True)
def moller_trumbore(surface, atom, direction):
    """
    Moller-Trumbore algorithm for finding the intersection between a ray and triangle

    Args:
        surface(np.ndarray): Array of triangles with shape [T, 3]
        positions(np.ndarray): Array of atom positions with shape [N, 3]
        direction(int): Axis normal to the plane that the surface lies in

    Return:
        p(np.ndarray): Array of coordinates for the intersection point with shape [N, 3]
        t(float): Vector of atom distances to surface with length N
    """
    # Vertices
    v0 = surface[:, 0, :]
    v1 = surface[:, 1, :]
    v2 = surface[:, 2, :]

    # Construct direction vector, which is unit normal orthogonal to surfaces
    d = np.zeros(3, dtype=np.float32)
    d[direction] = 1

    # Pre-compute edges
    e1 = v1 - v0
    e2 = v2 - v0

    # Pre-compute cross product between e2 and d
    q = np.cross(d, e2)

    # Pre-compute dot product between e1 and q
    a = (e1 * q).sum(-1)
    a_inv = 1 / a

    # Displace atom (origin) by v0
    s = atom - v0

    # Calculate U
    u = a_inv * (s * q).sum(-1)

    # Calculate V
    r = np.cross(s, e1)
    v = a_inv * (r * d).sum(-1)

    # Impose conditions (U, V > 0 and U + V <= 1)
    # Returns index of intersecting triangle
    idx = np.where((u >= 0) & (u < 1.0) & (v >= 0) & (v < 1.0) & (u + v <= 1))[0][0]

    # Calculate t (point distance from surface)
    t = a_inv[idx] * (e2[idx] * r[idx]).sum(-1)

    # Find intersection point
    p = atom + t * d

    return p, t


@jit(nopython=True)
def coarse_grain(dr, sigma, d):
    """
    Coarse graining function for the density field calculation

    Args:
        dr(np.ndarray): Vector of distances between a given atom and the grid points it contributes to
        sigma(float): Coarse graining width
        d(float): Dimensionality

    Return:
        Coarse grain density field
    """
    sigma2 = sigma ** 2
    dr2 = dr ** 2

    return (2 * np.pi * sigma2) ** (-d * 0.5) * np.exp(-dr2 / (2 * sigma2))
