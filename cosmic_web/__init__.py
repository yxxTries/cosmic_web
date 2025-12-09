"""
Cosmic web reconstruction tools.

Iteration 6:
- Cosmology definition.
- Coordinate transforms (RA, Dec, z) -> 3D comoving Cartesian coordinates.
- Geometry utilities for filament spines (segments and polylines).
- Filament abstraction to project galaxies onto a filament spine.
- DISPERSE I/O helpers to write tracer catalogs and read simple polyline skeletons.
- High-level mapping utilities to project galaxies onto a filament.
- Jackknife utilities and high-level pipeline helpers.
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
from .mapping import (
    GalaxyFilamentMapping,
    map_cartesian_to_filament,
    map_radec_z_to_filament,
)
from .jackknife import (
    JackknifeAssignment,
    JackknifeStats,
    assign_jackknife_grid_from_xyz,
    compute_jackknife_mean,
)
from .pipeline import (
    TracerCatalog,
    MappingWithFilament,
    prepare_tracer_catalog_from_radec_z,
    map_galaxies_to_filament_from_cartesian,
    map_galaxies_to_filament_from_radec_z,
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
    "GalaxyFilamentMapping",
    "map_cartesian_to_filament",
    "map_radec_z_to_filament",
    "JackknifeAssignment",
    "JackknifeStats",
    "assign_jackknife_grid_from_xyz",
    "compute_jackknife_mean",
    "TracerCatalog",
    "MappingWithFilament",
    "prepare_tracer_catalog_from_radec_z",
    "map_galaxies_to_filament_from_cartesian",
    "map_galaxies_to_filament_from_radec_z",
]
