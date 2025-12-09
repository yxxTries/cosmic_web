"""
Cosmic web reconstruction tools.

Iteration 4:
- Cosmology definition.
- Coordinate transforms (RA, Dec, z) -> 3D comoving Cartesian coordinates.
- Geometry utilities for filament spines (segments and polylines).
- Filament abstraction to project galaxies onto a filament spine.
- DISPERSE I/O helpers to write tracer catalogs and read simple polyline skeletons.
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
from .disperse import (
    write_tracers_ascii,
    load_polyline_vertices,
    filament_from_polyline_file,
    build_disperse_command,
)

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
    "write_tracers_ascii",
    "load_polyline_vertices",
    "filament_from_polyline_file",
    "build_disperse_command",
]
