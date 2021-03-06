# Isosurface
The isosurface code takes GROMACS MD simulation data to construct a 3D coase-grain density field to find instantaneous isosurfaces in solid-liquid two phase systems. This code has been written for use on biased simulations of bulk ice growth/melting as a way to distinguish true ice-like water from noisy/falsely tagged ice-like water as determined by the CHILL+ order parameter.

A 3D surface for every time frame is generated by using the marching cube algorithm on the density field. The marching cubes output can be interpolated to a fixed space grid for easy visualization in VMD as demonstrated below.

The Moller-Trumbore ray-triangle intersection algorithm is used to tag waters as either being solid or liquid depending which side of the interface the molecule lies on.

This code is also capable of detecting defects in the bulk liquid or ice by identifying all unique, fully-connected graphs from the marching cubes output.

## Installation
From the source code directory run:

```bash
python setup.py install
```

## Using the code
```python
from isosurface.isosurface import *

df = density_field(...)
```

## Isosurface Visualization
The following show animations of the isosurface generated by the code and visualized in VMD. To visualize, the marching cubes triangular surface mesh is interpolated to a rectangular grid with a fixed number of nodes for all time steps. These nodes are written to a pdb file that VMD is capable of loading. The coloring scheme shows the height of a node relative to the average height of all nodes in the surface. The second animation shows the mobile ice molecules where the white molecules are true positive ice and the black flecks are false positive ice molecules determined by the CHILL+ algorithm.

![iso](https://user-images.githubusercontent.com/31362150/138544559-10d63389-6ba4-4804-a677-335c1d8c62c3.gif) ![iso_with_ice](https://user-images.githubusercontent.com/31362150/138544562-65898426-0235-4b39-aabf-725de9ab3cd7.gif)

