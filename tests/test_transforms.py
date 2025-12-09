import numpy as np

from cosmic_web.cosmology import comoving_distance_mpc
from cosmic_web.coords import radec_z_to_cartesian, radec_z_to_cartesian_single


def test_radec_z_to_cartesian_shape_scalar():
    coords = radec_z_to_cartesian(0.0, 0.0, 0.03)
    assert coords.shape == (1, 3)


def test_radec_z_to_cartesian_shape_vector():
    ra = [0.0, 90.0, 0.0]
    dec = [0.0, 0.0, 90.0]
    z = [0.03, 0.03, 0.03]

    coords = radec_z_to_cartesian(ra, dec, z)
    assert coords.shape == (3, 3)


def test_cartesian_radius_matches_comoving_distance():
    ra = [0.0, 45.0, 180.0]
    dec = [-30.0, 0.0, 60.0]
    z = [0.01, 0.02, 0.03]

    coords = radec_z_to_cartesian(ra, dec, z)
    d_c = comoving_distance_mpc(z)
    radii = np.linalg.norm(coords, axis=1)

    assert np.allclose(radii, d_c, rtol=1e-10, atol=0.0)


def test_special_directions_on_axes():
    z_val = 0.03
    d_c = comoving_distance_mpc(z_val)

    x, y, z_cart = radec_z_to_cartesian_single(0.0, 0.0, z_val)
    assert np.isclose(x, d_c, rtol=1e-10, atol=0.0)
    assert np.isclose(y, 0.0, atol=1e-10)
    assert np.isclose(z_cart, 0.0, atol=1e-10)

    x2, y2, z2 = radec_z_to_cartesian_single(90.0, 0.0, z_val)
    assert np.isclose(y2, d_c, rtol=1e-10, atol=0.0)
    assert np.isclose(x2, 0.0, atol=1e-10)
    assert np.isclose(z2, 0.0, atol=1e-10)

    x3, y3, z3 = radec_z_to_cartesian_single(0.0, 90.0, z_val)
    assert np.isclose(z3, d_c, rtol=1e-10, atol=0.0)
    assert np.isclose(x3, 0.0, atol=1e-10)
    assert np.isclose(y3, 0.0, atol=1e-10)
