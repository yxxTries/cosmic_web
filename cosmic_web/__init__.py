"""
Cosmic web reconstruction tools.

Iteration 3:
- Cosmology definition
- Coordinate transforms (RA, Dec, z) -> 3D comoving Cartesian coordinates.
- Geometry utilities for filament spines (segments and polylines).
- Filament abstraction to project galaxies onto a filament spine.
"""

from .cosmology import COSMO, comoving_distance_mpc
from .coords import radec_z_to_cartesian, radec_z_to_cartesian_single
from .geometry import (
    compute_arc_lengths,
    project_point_onto_segment,
    project_point_onto_polyline,
    ProjectionResult,
)
from .filament import Filament

__all__ = [
    "COSMO",
    "comoving_distance_mpc",
    "radec_z_to_cartesian",
    "radec_z_to_cartesian_single",
    "compute_arc_lengths",
    "project_point_onto_segment",
    "project_point_onto_polyline",
    "ProjectionResult",
    "Filament",
]
