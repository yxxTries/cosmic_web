import numpy as np

from cosmic_web.pipeline import (
    TracerCatalog,
    MappingWithFilament,
    prepare_tracer_catalog_from_radec_z,
    map_galaxies_to_filament_from_cartesian,
    map_galaxies_to_filament_from_radec_z,
)


def test_prepare_tracer_catalog_from_radec_z(tmp_path):
    ra = [0.0, 90.0]
    dec = [0.0, 0.0]
    z = [0.01, 0.02]

    out_path = tmp_path / "tracers.dat"
    catalog = prepare_tracer_catalog_from_radec_z(ra, dec, z, out_path)

    assert isinstance(catalog, TracerCatalog)
    assert catalog.coords_xyz.shape == (2, 3)
    assert catalog.path == out_path
    assert out_path.exists()

    loaded = np.loadtxt(out_path)
    if loaded.ndim == 1:
        loaded = loaded.reshape(1, -1)
    assert loaded.shape == (2, 3)


def test_map_galaxies_to_filament_from_cartesian(tmp_path):
    # Simple filament along x-axis
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    poly_path = tmp_path / "filament.txt"
    np.savetxt(poly_path, vertices, fmt="%.8f")

    coords = np.array(
        [
            [5.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )

    result = map_galaxies_to_filament_from_cartesian(coords, poly_path)

    assert isinstance(result, MappingWithFilament)
    assert result.mapping.coords_xyz.shape == (2, 3)
    assert result.mapping.distances.shape == (2,)
    assert result.mapping.s_along.shape == (2,)


def test_map_galaxies_to_filament_from_radec_z(tmp_path):
    # Filament along +x-axis
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
        ]
    )
    poly_path = tmp_path / "filament.txt"
    np.savetxt(poly_path, vertices, fmt="%.8f")

    ra = [0.0, 0.0]
    dec = [0.0, 0.0]
    z = [0.01, 0.02]

    result = map_galaxies_to_filament_from_radec_z(ra, dec, z, poly_path)

    assert isinstance(result, MappingWithFilament)
    assert result.mapping.coords_xyz.shape == (2, 3)
    assert result.mapping.distances.shape == (2,)
    assert np.all(result.mapping.distances < 1e-6)
