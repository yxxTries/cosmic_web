from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from cosmic_web.cosmology import comoving_distance_mpc


def _to_rad(deg: np.ndarray) -> np.ndarray:
    """Convert degrees to radians."""
    return np.deg2rad(deg.astype(float))


def radec_z_to_cartesian(
    ra_deg: float | Iterable[float] | np.ndarray,
    dec_deg: float | Iterable[float] | np.ndarray,
    z: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """
    Convert (RA, Dec, z) to 3D comoving Cartesian coordinates (x, y, z) in Mpc.

    Convention:
        x = D_c * cos(dec) * cos(ra)
        y = D_c * cos(dec) * sin(ra)
        z = D_c * sin(dec)

    where D_c is the comoving distance in Mpc.

    Parameters
    ----------
    ra_deg:
        Right ascension in degrees. Scalar or array-like.
    dec_deg:
        Declination in degrees. Scalar or array-like.
    z:
        Redshift. Scalar or array-like.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing [x, y, z] in Mpc.
        If inputs are scalars, the result has shape (1, 3).
    """
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    ra_b, dec_b, z_b = np.broadcast_arrays(ra_arr, dec_arr, z_arr)

    ra_rad = _to_rad(ra_b)
    dec_rad = _to_rad(dec_b)

    d_c = comoving_distance_mpc(z_b)

    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_ra = np.cos(ra_rad)
    sin_ra = np.sin(ra_rad)

    x = d_c * cos_dec * cos_ra
    y = d_c * cos_dec * sin_ra
    z_cart = d_c * sin_dec

    coords = np.stack([x, y, z_cart], axis=-1)

    if coords.ndim == 1:
        coords = coords.reshape(1, 3)

    return coords


def radec_z_to_cartesian_single(
    ra_deg: float,
    dec_deg: float,
    z: float,
) -> Tuple[float, float, float]:
    """
    Convenience wrapper for a single (RA, Dec, z) triplet.

    Returns
    -------
    (x, y, z): tuple of floats
        Comoving coordinates in Mpc.
    """
    coords = radec_z_to_cartesian(ra_deg, dec_deg, z)
    x, y, z_cart = coords[0]
    return float(x), float(y), float(z_cart)
