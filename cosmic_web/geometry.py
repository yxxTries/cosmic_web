from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    """
    Result of projecting a 3D point onto a polyline (filament spine).

    Attributes
    ----------
    distance:
        Minimum Euclidean distance from the point to the polyline.
    closest_point:
        3D coordinates of the closest point on the polyline.
    segment_index:
        Index of the segment on which the closest point lies (0-based).
    t:
        Parametric coordinate along that segment in [0, 1].
    s_along:
        Arc-length coordinate along the polyline at the closest point (Mpc).
    """
    distance: float
    closest_point: np.ndarray
    segment_index: int
    t: float
    s_along: float


def _validate_vertices(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if v.shape[0] < 2:
        raise ValueError("polyline requires at least 2 vertices")
    return v


def compute_arc_lengths(vertices: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc-lengths along a 3D polyline.

    Parameters
    ----------
    vertices:
        Array of shape (N, 3) representing the ordered vertices of the polyline.

    Returns
    -------
    np.ndarray
        Array of shape (N,) where element i is the arc-length from the first
        vertex up to vertex i. The first element is always 0.0.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if v.shape[0] == 0:
        raise ValueError("at least one vertex required")

    if v.shape[0] == 1:
        return np.array([0.0], dtype=float)

    diffs = np.diff(v, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.empty(v.shape[0], dtype=float)
    arc[0] = 0.0
    arc[1:] = np.cumsum(seg_lengths)
    return arc


def project_point_onto_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """
    Project a 3D point onto a line segment.

    Parameters
    ----------
    point:
        3D point as array-like of shape (3,).
    start, end:
        Segment endpoints as array-like of shape (3,).

    Returns
    -------
    distance:
        Euclidean distance from point to the closest point on the segment.
    closest_point:
        3D coordinates of the closest point on the segment.
    t:
        Parametric coordinate in [0, 1] along the segment:
        closest_point = start + t * (end - start).
        If the segment is degenerate (start == end), t is 0.0.
    """
    p = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(start, dtype=float).reshape(3)
    b = np.asarray(end, dtype=float).reshape(3)

    ab = b - a
    ab2 = float(np.dot(ab, ab))

    if ab2 == 0.0:
        # Degenerate segment: treat as a single point.
        closest = a
        dist = float(np.linalg.norm(p - closest))
        return dist, closest, 0.0

    t = float(np.dot(p - a, ab) / ab2)
    t_clamped = max(0.0, min(1.0, t))

    closest = a + t_clamped * ab
    dist = float(np.linalg.norm(p - closest))
    return dist, closest, t_clamped


def project_point_onto_polyline(
    point: np.ndarray,
    vertices: np.ndarray,
) -> ProjectionResult:
    """
    Project a 3D point onto a polyline defined by ordered vertices.

    Parameters
    ----------
    point:
        3D point as array-like of shape (3,).
    vertices:
        Array of shape (N, 3) representing the polyline vertices.

    Returns
    -------
    ProjectionResult
        Detailed information about the closest point on the polyline.
    """
    p = np.asarray(point, dtype=float).reshape(3)
    v = _validate_vertices(vertices)
    arc = compute_arc_lengths(v)

    best_distance = np.inf
    best_point = None
    best_seg_index = -1
    best_t = 0.0

    diffs = np.diff(v, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    for i in range(v.shape[0] - 1):
        start = v[i]
        end = v[i + 1]
        distance, closest, t = project_point_onto_segment(p, start, end)
        if distance < best_distance:
            best_distance = distance
            best_point = closest
            best_seg_index = i
            best_t = t

    seg_len = seg_lengths[best_seg_index]
    s_along = float(arc[best_seg_index] + best_t * seg_len)

    return ProjectionResult(
        distance=float(best_distance),
        closest_point=np.asarray(best_point, dtype=float),
        segment_index=int(best_seg_index),
        t=float(best_t),
        s_along=s_along,
    )
