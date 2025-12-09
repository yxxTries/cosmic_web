from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class JackknifeAssignment:
    """
    Jackknife region assignment for a set of objects.

    Attributes
    ----------
    region_ids:
        (N,) array of integer region indices in [0, n_regions-1].
    n_regions:
        Total number of jackknife regions.
    """
    region_ids: np.ndarray
    n_regions: int


@dataclass(frozen=True)
class JackknifeStats:
    """
    Jackknife statistics for a scalar estimator (e.g. mean distance to filament).

    Attributes
    ----------
    theta_full:
        Estimator evaluated on the full sample.
    theta_jackknife:
        Jackknife estimate (mean of leave-one-region-out estimates).
    variance:
        Jackknife variance estimate.
    stderr:
        Square root of the variance.
    """
    theta_full: float
    theta_jackknife: float
    variance: float
    stderr: float


def assign_jackknife_grid_from_xyz(
    coords_xyz: Iterable[Iterable[float]],
    n_regions: int,
    axes: Tuple[int, int] = (0, 1),
) -> JackknifeAssignment:
    """
    Assign jackknife regions using a regular 2D grid in the chosen axes.

    This is a simple geometric partition intended as a building block. In a real
    survey analysis you would adapt this to the survey mask.

    Parameters
    ----------
    coords_xyz:
        (N, 3) array of Cartesian coordinates.
    n_regions:
        Desired number of jackknife regions.
    axes:
        Tuple of two axes (0, 1, or 2) to use for the 2D grid (default: x-y).

    Returns
    -------
    JackknifeAssignment
    """
    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_xyz must have shape (N, 3)")
    if n_regions <= 0:
        raise ValueError("n_regions must be positive")

    ax0, ax1 = axes
    if ax0 not in (0, 1, 2) or ax1 not in (0, 1, 2):
        raise ValueError("axes must be a tuple of two integers in {0,1,2}")
    if ax0 == ax1:
        raise ValueError("axes must be distinct")

    x = coords[:, ax0]
    y = coords[:, ax1]

    # Choose grid dimensions approximately square
    nx = int(np.floor(np.sqrt(n_regions)))
    ny = int(np.ceil(n_regions / nx))

    # Bounding box
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Avoid zero-width ranges
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    # Map coordinates to grid indices
    ix = ((x - x_min) / (x_max - x_min) * nx).astype(int)
    iy = ((y - y_min) / (y_max - y_min) * ny).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    region_ids = ix + nx * iy

    # If total number of grid cells exceeds requested n_regions, merge highest indices
    region_ids = np.minimum(region_ids, n_regions - 1)

    return JackknifeAssignment(region_ids=region_ids, n_regions=n_regions)


def compute_jackknife_mean(
    values: Iterable[float],
    assignment: JackknifeAssignment,
) -> JackknifeStats:
    """
    Compute jackknife statistics for the mean of a set of scalar values.

    This follows the standard delete-one-group jackknife:

        theta_k = estimator on sample with region k left out
        theta_bar = mean(theta_k)
        Var(theta) = (R - 1) / R * sum_k (theta_k - theta_bar)^2

    where R is the number of regions.

    Parameters
    ----------
    values:
        Iterable of scalar measurements (e.g. distances to filament).
    assignment:
        JackknifeAssignment describing region membership.

    Returns
    -------
    JackknifeStats
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1:
        raise ValueError("values must be a 1D array")
    region_ids = np.asarray(assignment.region_ids, dtype=int)
    if region_ids.shape[0] != vals.shape[0]:
        raise ValueError("values and region_ids must have the same length")

    R = int(assignment.n_regions)
    if R <= 1:
        raise ValueError("at least two jackknife regions are required")

    theta_full = float(np.mean(vals))

    theta_k = np.empty(R, dtype=float)
    for k in range(R):
        mask = region_ids != k
        if not np.any(mask):
            raise ValueError(f"region {k} is empty; cannot form jackknife sample")
        theta_k[k] = float(np.mean(vals[mask]))

    theta_bar = float(np.mean(theta_k))
    variance = float((R - 1) / R * np.sum((theta_k - theta_bar) ** 2))
    stderr = float(np.sqrt(variance))

    return JackknifeStats(
        theta_full=theta_full,
        theta_jackknife=theta_bar,
        variance=variance,
        stderr=stderr,
    )
