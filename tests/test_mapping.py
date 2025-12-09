import numpy as np

from cosmic_web.filament import Filament
from cosmic_web.mapping import (
    GalaxyFilamentMapping,
    map_cartesian_to_filament,
    map_radec_z_to_filament,
)
from cosmic_web.cosmology import comoving_distance_mpc


def test_map_cartesian_to_filament_straight_line():
    # Filament along x-axis from 0 to 10
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    filament = Filament.from_vertices(vertices)

    coords = np.array(
        [
            [5.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )

    mapping = map_cartesian_to_filament(coords, filament)

    assert isinstance(mapping, GalaxyFilamentMapping)
    assert mapping.coords_xyz.shape == (2, 3)
    assert mapping.distances.shape == (2,)
    assert mapping.s_along.shape == (2,)

    # First galaxy: perpendicular projection at x=5
    assert np.isclose(mapping.distances[0], 1.0)
    assert np.isclose(mapping.s_along[0], 5.0)

    # Second galaxy: closest to first vertex
    assert np.isclose(mapping.distances[1], 2.0)
    assert np.isclose(mapping.s_along[1], 0.0)


def test_map_radec_z_to_filament_on_axis():
    # Filament along +x-axis at some comoving distance
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    filament = Filament.from_vertices(vertices)

    # RA=0, Dec=0: points along +x axis
    z_vals = np.array([0.01, 0.02])
    d_c = comoving_distance_mpc(z_vals)

    mapping = map_radec_z_to_filament(
        ra_deg=[0.0, 0.0],
        dec_deg=[0.0, 0.0],
        z=z_vals,
        filament=filament,
    )

    # On-axis: distance to filament should be ~0, s_along ~ comoving distance (clamped by filament length)
    assert np.all(mapping.distances >= 0.0)
    assert np.all(mapping.distances < 1e-6)

    # s_along should be close to comoving distance but cannot exceed filament length
    assert np.all(mapping.s_along <= filament.length + 1e-6)
