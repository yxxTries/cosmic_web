from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .geometry import compute_arc_lengths, project_point_onto_polyline, ProjectionResult


@dataclass(frozen=True)
class Filament:
    """
    Representation of a single filament spine as a 3D polyline.

    Attributes
    ----------
    vertices:
        Array of shape (N, 3) with the ordered 3D vertices of the spine.
    arc_lengths:
        Array of shape (N,) with cumulative arc-lengths (Mpc) from the first vertex.
    """
    vertices: np.ndarray
    arc_lengths: np.ndarray

    @classmethod
    def from_vertices(cls, vertices: Iterable[Iterable[float]]) -> "Filament":
        """
        Construct a Filament from an iterable of vertices.

        Parameters
        ----------
        vertices:
            Iterable of (x, y, z) coordinates.

        Returns
        -------
        Filament
        """
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if v.shape[0] < 2:
            raise ValueError("filament requires at least 2 vertices")
        arc = compute_arc_lengths(v)
        return cls(vertices=v, arc_lengths=arc)

    @property
    def length(self) -> float:
        """Total length of the filament spine in Mpc."""
        return float(self.arc_lengths[-1])

    def project_point(self, point: Iterable[float]) -> ProjectionResult:
        """
        Project a 3D point onto this filament spine.

        Parameters
        ----------
        point:
            3D point (x, y, z).

        Returns
        -------
        ProjectionResult
            Contains distance-to-filament and arc-length along the spine.
        """
        p = np.asarray(point, dtype=float)
        return project_point_onto_polyline(p, self.vertices)

    def project_points(self, points: Iterable[Iterable[float]]) -> List[ProjectionResult]:
        """
        Project multiple 3D points onto this filament spine.

        Parameters
        ----------
        points:
            Iterable of 3D points.

        Returns
        -------
        list[ProjectionResult]
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        results: List[ProjectionResult] = []
        for p in pts:
            results.append(self.project_point(p))
        return results

    def distances_and_s_along(self, points: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to get arrays of distance-to-filament and s_along.

        Parameters
        ----------
        points:
            Iterable of 3D points.

        Returns
        -------
        distances: np.ndarray
            1D array of minimum distances to the filament spine (Mpc).
        s_along: np.ndarray
            1D array of arc-length coordinates along the spine at the closest points (Mpc).
        """
        results = self.project_points(points)
        distances = np.array([r.distance for r in results], dtype=float)
        s_values = np.array([r.s_along for r in results], dtype=float)
        return distances, s_values
