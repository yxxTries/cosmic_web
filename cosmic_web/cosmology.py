from __future__ import annotations

from typing import Iterable

import numpy as np
from astropy.cosmology import FlatLambdaCDM


# Cosmology matching the paper:
# H0 = 70 km/s/Mpc, Omega_M = 0.3, Omega_Lambda = 0.7
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


def comoving_distance_mpc(z: float | np.ndarray | Iterable[float]) -> np.ndarray:
    """
    Compute the line-of-sight comoving distance in Mpc for one or more redshifts.

    Parameters
    ----------
    z:
        Scalar or array-like of redshifts.

    Returns
    -------
    np.ndarray
        Comoving distances in Mpc with the same shape as `z`.
    """
    z_arr = np.asarray(z, dtype=float)
    dc = COSMO.comoving_distance(z_arr)
    return np.asarray(dc.value, dtype=float)
