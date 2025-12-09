import numpy as np

from cosmic_web.filament import Filament


def test_filament_length_and_arc_lengths():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )
    filament = Filament.from_vertices(vertices)

    # Length is 3 + 4 = 7
    assert np.isclose(filament.length, 7.0)
    assert filament.arc_lengths.shape == (3,)
    assert np.allclose(filament.arc_lengths, np.array([0.0, 3.0, 7.0]))


def test_filament_project_point_straight_line():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    filament = Filament.from_vertices(vertices)

    point = np.array([5.0, 1.0, 0.0])
    result = filament.project_point(point)

    assert np.isclose(result.distance, 1.0)
    assert np.allclose(result.closest_point, np.array([5.0, 0.0, 0.0]))
    assert np.isclose(result.s_along, 5.0)


def test_filament_distances_and_s_along_multiple_points():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    filament = Filament.from_vertices(vertices)

    points = np.array(
        [
            [5.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )

    distances, s_along = filament.distances_and_s_along(points)

    assert distances.shape == (2,)
    assert s_along.shape == (2,)

    # First point: perpendicular at x=5, distance=1, s_along=5
    assert np.isclose(distances[0], 1.0)
    assert np.isclose(s_along[0], 5.0)

    # Second point: closest to first vertex at x=0, distance=2, s_along=0
    assert np.isclose(distances[1], 2.0)
    assert np.isclose(s_along[1], 0.0)
