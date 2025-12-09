import numpy as np

from cosmic_web.disperse import (
    write_tracers_ascii,
    load_polyline_vertices,
    filament_from_polyline_file,
    build_disperse_command,
)


def test_write_tracers_and_load_polyline(tmp_path):
    coords = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]
    )
    out = tmp_path / "tracers.txt"
    write_tracers_ascii(out, coords)

    loaded = load_polyline_vertices(out)
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded, coords)


def test_filament_from_polyline_file(tmp_path):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    poly = tmp_path / "poly.txt"
    np.savetxt(poly, vertices, fmt="%.8f")

    filament = filament_from_polyline_file(poly)
    assert np.isclose(filament.length, 10.0)
    assert filament.vertices.shape == (2, 3)


def test_build_disperse_command_basic():
    cmd = build_disperse_command("tracers.dat", "skeleton", nsig=5.0, mirror_boundary=True, dimensionality=3)

    assert cmd[0] == "disperse"
    assert "tracers.dat" in cmd
    assert "-3D" in cmd
    assert "-nsig" in cmd
    assert "5.00" in cmd
    assert "-b" in cmd and "m" in cmd
    assert "-o" in cmd and "skeleton" in cmd
