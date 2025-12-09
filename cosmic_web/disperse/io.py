from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from cosmic_web.filament import Filament


def write_tracers_ascii(path: str | Path, coords: Iterable[Iterable[float]]) -> None:
    """
    Write a tracer catalog (e.g. galaxy positions) to an ASCII file for DISPERSE.

    The output format is one point per line:
        x  y  z

    Parameters
    ----------
    path:
        Output file path.
    coords:
        Iterable of (x, y, z) coordinates in Mpc.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(path, arr, fmt="%.8f")


def load_polyline_vertices(path: str | Path) -> np.ndarray:
    """
    Load a simple polyline skeleton from an ASCII file.

    Expected format: one vertex per line with 3 columns:
        x  y  z

    Parameters
    ----------
    path:
        Input file path.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3).
    """
    path = Path(path)
    arr = np.loadtxt(path, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (x y z), got shape {arr.shape}")

    return arr


def filament_from_polyline_file(path: str | Path) -> Filament:
    """
    Convenience helper to construct a Filament from a simple ASCII polyline file.

    Parameters
    ----------
    path:
        Input file path with 3-column (x, y, z) vertices.

    Returns
    -------
    Filament
    """
    vertices = load_polyline_vertices(path)
    return Filament.from_vertices(vertices)


def build_disperse_command(
    tracer_path: str | Path,
    output_prefix: str | Path,
    nsig: float = 5.0,
    mirror_boundary: bool = True,
    dimensionality: int = 3,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Build the command-line argument list to run DISPERSE.

    This does not execute DISPERSE; it only returns the argument list so you
    can pass it to subprocess.run() yourself.

    Parameters
    ----------
    tracer_path:
        Path to the tracer catalog file (ASCII).
    output_prefix:
        Prefix for DISPERSE output files.
    nsig:
        Significance threshold (e.g. 5.0 for 5-sigma).
    mirror_boundary:
        If True, use mirror boundary conditions ("-b m").
    dimensionality:
        3 for 3D analysis, 2 for 2D (used to pick the DISPERSE flag).
    extra_args:
        Optional sequence of additional command-line arguments.

    Returns
    -------
    list[str]
        The full command-line argument list, starting with "disperse".
    """
    tracer_path = Path(tracer_path)
    output_prefix = Path(output_prefix)

    if dimensionality == 3:
        dim_flag = "-3D"
    elif dimensionality == 2:
        dim_flag = "-2D"
    else:
        raise ValueError("dimensionality must be 2 or 3")

    cmd: List[str] = [
        "disperse",
        str(tracer_path),
        dim_flag,
        "-nsig",
        f"{nsig:.2f}",
    ]

    if mirror_boundary:
        cmd.extend(["-b", "m"])

    cmd.extend(["-o", str(output_prefix)])

    if extra_args:
        cmd.extend(list(extra_args))

    return cmd
