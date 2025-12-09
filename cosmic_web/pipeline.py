from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .coords import radec_z_to_cartesian
from .disperse import write_tracers_ascii, filament_from_polyline_file
from .filament import Filament
from .mapping import GalaxyFilamentMapping, map_cartesian_to_filament, map_radec_z_to_filament


@dataclass(frozen=True)
class TracerCatalog:
    """
    Tracer catalog ready for DISPERSE.

    Attributes
    ----------
    coords_xyz:
        (N, 3) array of tracer Cartesian coordinates in Mpc.
    path:
        Path to the ASCII tracer file on disk.
    """
    coords_xyz: np.ndarray
    path: Path


@dataclass(frozen=True)
class MappingWithFilament:
    """
    Convenience container bundling a Filament with its galaxy mapping.

    Attributes
    ----------
    filament:
        Filament instance.
    mapping:
        GalaxyFilamentMapping instance.
    """
    filament: Filament
    mapping: GalaxyFilamentMapping


def prepare_tracer_catalog_from_radec_z(
    ra_deg: Iterable[float] | np.ndarray,
    dec_deg: Iterable[float] | np.ndarray,
    z: Iterable[float] | np.ndarray,
    output_path: str | Path,
) -> TracerCatalog:
    """
    Convert (RA, Dec, z) galaxy positions to Cartesian coordinates and write
    an ASCII tracer catalog for DISPERSE.

    Parameters
    ----------
    ra_deg, dec_deg, z:
        Galaxy sky coordinates and redshifts.
    output_path:
        Path to the output ASCII file.

    Returns
    -------
    TracerCatalog
    """
    coords_xyz = radec_z_to_cartesian(ra_deg, dec_deg, z)
    output_path = Path(output_path)
    write_tracers_ascii(output_path, coords_xyz)
    return TracerCatalog(coords_xyz=coords_xyz, path=output_path)


def map_galaxies_to_filament_from_cartesian(
    coords_xyz: Iterable[Iterable[float]],
    filament_polyline_path: str | Path,
) -> MappingWithFilament:
    """
    Convenience helper: load a filament from a polyline file and map galaxies
    (given in Cartesian coordinates) onto it.

    Parameters
    ----------
    coords_xyz:
        (N, 3) array of galaxy positions in Mpc.
    filament_polyline_path:
        Path to ASCII file with filament polyline vertices.

    Returns
    -------
    MappingWithFilament
    """
    filament = filament_from_polyline_file(filament_polyline_path)
    mapping = map_cartesian_to_filament(coords_xyz, filament)
    return MappingWithFilament(filament=filament, mapping=mapping)


def map_galaxies_to_filament_from_radec_z(
    ra_deg: Iterable[float] | np.ndarray,
    dec_deg: Iterable[float] | np.ndarray,
    z: Iterable[float] | np.ndarray,
    filament_polyline_path: str | Path,
) -> MappingWithFilament:
    """
    Convenience helper: load a filament from a polyline file and map galaxies
    from (RA, Dec, z) onto it.

    Parameters
    ----------
    ra_deg, dec_deg, z:
        Galaxy sky coordinates and redshifts.
    filament_polyline_path:
        Path to ASCII file with filament polyline vertices.

    Returns
    -------
    MappingWithFilament
    """
    filament = filament_from_polyline_file(filament_polyline_path)
    mapping = map_radec_z_to_filament(ra_deg, dec_deg, z, filament)
    return MappingWithFilament(filament=filament, mapping=mapping)
