import numpy as np

from cosmic_web.jackknife import (
    JackknifeAssignment,
    JackknifeStats,
    assign_jackknife_grid_from_xyz,
    compute_jackknife_mean,
)


def test_assign_jackknife_grid_from_xyz_simple_grid():
    # Four points forming a square in x-y; request 4 regions
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    assignment = assign_jackknife_grid_from_xyz(coords, n_regions=4, axes=(0, 1))

    assert assignment.n_regions == 4
    assert assignment.region_ids.shape == (4,)
    # All region ids should be between 0 and 3
    assert np.all(assignment.region_ids >= 0)
    assert np.all(assignment.region_ids < 4)
    # There should be at least 2 distinct regions in this simple test
    assert np.unique(assignment.region_ids).size >= 2


def test_compute_jackknife_mean_reproduces_mean_for_linear_estimator():
    # Values assigned to 2 regions
    values = np.array([1.0, 2.0, 3.0, 4.0])
    region_ids = np.array([0, 0, 1, 1])
    assignment = JackknifeAssignment(region_ids=region_ids, n_regions=2)

    stats = compute_jackknife_mean(values, assignment)

    full_mean = np.mean(values)
    assert isinstance(stats, JackknifeStats)
    assert np.isclose(stats.theta_full, full_mean)
    # For the mean, the jackknife estimate should equal the full mean exactly
    assert np.isclose(stats.theta_jackknife, full_mean)
    # Variance should be positive
    assert stats.variance >= 0.0
