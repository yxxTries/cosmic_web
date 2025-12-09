from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .coords import radec_z_to_cartesian
from .filament import Filament


@dataclass(frozen=True)
class GalaxyFilamentMapping:
    """
    Container for the result of projecting a set of galaxies onto a filament.

    Attributes
    ----------
    coords_xyz:
        (N, 3) array of galaxy Cartesian coordinates in Mpc.
    distances:
        (N,) array of minimum distances to the filament spine in Mpc.
    s_along:
        (N,) array of arc-length coordinates along the spine in Mpc.
    """
    coords_xyz: np.ndarray
    distances: np.ndarray
    s_along: np.ndarray


def map_cartesian_to_filament(
    coords_xyz: Iterable[Iterable[float]],
    filament: Filament,
) -> GalaxyFilamentMapping:
    """
    Project galaxies given in Cartesian coordinates onto a filament.

    Parameters
    ----------
    coords_xyz:
        Iterable of (x, y, z) galaxy positions in Mpc.
    filament:
        Filament instance representing the spine.

    Returns
    -------
    GalaxyFilamentMapping
    """
    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    if coords.shape[1] != 3:
        raise ValueError("coords_xyz must have shape (N, 3)")

    distances, s_along = filament.distances_and_s_along(coords)
    return GalaxyFilamentMapping(coords_xyz=coords, distances=distances, s_along=s_along)


def map_radec_z_to_filament(
    ra_deg: Iterable[float] | np.ndarray,
    dec_deg: Iterable[float] | np.ndarray,
    z: Iterable[float] | np.ndarray,
    filament: Filament,
) -> GalaxyFilamentMapping:
    """
    Convert (RA, Dec, z) to Cartesian coordinates and project onto a filament.

    Parameters
    ----------
    ra_deg, dec_deg, z:
        Galaxy sky coordinates and redshifts.
    filament:
        Filament instance representing the spine.

    Returns
    -------
    GalaxyFilamentMapping
    """
    coords = radec_z_to_cartesian(ra_deg, dec_deg, z)
    return map_cartesian_to_filament(coords, filament)
