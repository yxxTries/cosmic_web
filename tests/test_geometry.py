import numpy as np

from cosmic_web.geometry import (
    compute_arc_lengths,
    project_point_onto_segment,
    project_point_onto_polyline,
)


def test_compute_arc_lengths_simple_polyline():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )
    arc = compute_arc_lengths(vertices)
    expected = np.array([0.0, 3.0, 7.0])
    assert arc.shape == (3,)
    assert np.allclose(arc, expected)


def test_project_point_onto_segment_degenerate():
    point = np.array([1.0, 2.0, 5.0])
    start = np.array([1.0, 2.0, 3.0])
    end = np.array([1.0, 2.0, 3.0])  # degenerate segment

    dist, closest, t = project_point_onto_segment(point, start, end)
    assert np.isclose(dist, 2.0)
    assert np.allclose(closest, start)
    assert np.isclose(t, 0.0)


def test_project_point_onto_polyline_straight_line():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    point = np.array([5.0, 1.0, 0.0])

    result = project_point_onto_polyline(point, vertices)

    assert np.isclose(result.distance, 1.0)
    assert np.allclose(result.closest_point, np.array([5.0, 0.0, 0.0]))
    assert result.segment_index == 0
    assert np.isclose(result.t, 0.5)
    assert np.isclose(result.s_along, 5.0)


def test_project_point_onto_polyline_beyond_endpoints():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    # Left of the first vertex
    point_left = np.array([-1.0, 0.0, 0.0])
    res_left = project_point_onto_polyline(point_left, vertices)
    assert np.isclose(res_left.distance, 1.0)
    assert np.allclose(res_left.closest_point, np.array([0.0, 0.0, 0.0]))
    assert np.isclose(res_left.s_along, 0.0)

    # Right of the last vertex
    point_right = np.array([12.0, 0.0, 0.0])
    res_right = project_point_onto_polyline(point_right, vertices)
    assert np.isclose(res_right.distance, 2.0)
    assert np.allclose(res_right.closest_point, np.array([10.0, 0.0, 0.0]))
    assert np.isclose(res_right.s_along, 10.0)


def test_project_point_onto_polyline_slanted_segment():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    point = np.array([0.0, 3.0, 3.0])

    result = project_point_onto_polyline(point, vertices)

    assert np.isclose(result.distance, 3.0)
    assert np.allclose(result.closest_point, np.array([0.0, 0.0, 3.0]))
    assert np.isclose(result.s_along, 3.0)
    assert np.isclose(result.t, 0.3)
