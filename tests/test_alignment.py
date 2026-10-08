import numpy as np
import pytest

from granular_analysis.analysis.alignment import particle_alignment
from granular_analysis.analysis.alignment import enrich_alignment
from granular_analysis.scripts.run_all import build_parser


def particle(**changes):
    return dict(dict(u=[1, 0, 0], a=2, b=1, c=1, q=[1, 0, 0, 0], t=3), **changes)


@pytest.mark.parametrize("velocity,angle", [([1, 0, 0], 0), ([-1, 0, 0], 0),
                                            ([0, 1, 0], 90), ([1, 1, 0], 45)])
def test_prolate(velocity, angle):
    assert particle_alignment([particle(u=velocity)]).angle_deg[0] == pytest.approx(angle)


def test_rotated_axis_and_no_mutation():
    q = np.array([1., 0, 0, 1.])
    result = particle_alignment([particle(q=q, u=[0, 1, 0])])
    assert result.angle_deg[0] == pytest.approx(0, abs=2e-6)
    np.testing.assert_array_equal(q, [1, 0, 0, 1])


def test_largest_axis_not_always_a():
    result = particle_alignment([particle(a=1, b=3, u=[0, 1, 0])])
    assert result.axis[0] == "b"
    assert result.angle_deg[0] == 0


@pytest.mark.parametrize("velocity,angle", [([0, 0, 1], 0), ([0, 0, -1], 0), ([1, 0, 0], 90), ([1, 0, 1], 45)])
def test_oblate(velocity, angle):
    assert particle_alignment([particle(b=2, u=velocity)], "oblate").angle_deg[0] == pytest.approx(angle)


def test_oblate_components_use_direct_angle():
    particles = [particle(b=2, u=[0, 0, 2]), particle(b=2, u=[3, 0, 0])]
    table, metadata = enrich_alignment(particle_alignment(particles, "oblate"), particles)
    np.testing.assert_allclose(table.abs_cos_angle, [1, 0], atol=1e-15)
    np.testing.assert_allclose(table.v_axis, [2, 0], atol=1e-15)
    np.testing.assert_allclose(table.v_plane, [0, 3], atol=1e-15)
    np.testing.assert_allclose(metadata["feature_mean"], [45, 2.5])


def test_undefined_angles_and_threshold():
    result = particle_alignment([particle(u=[0, 0, 0]), particle(b=2), particle(u=[0.01, 0, 0])], min_speed=.1)
    assert result.angle_deg.isna().all()
    assert list(result.status) == ["stationary", "ambiguous_longest_axis", "stationary"]
    assert list(result.particle_row) == [1, 2, 3]


def test_invalid_quaternion():
    with pytest.raises(ValueError, match="quaternion"):
        particle_alignment([particle(q=[0, 0, 0, 0])])


def test_cli():
    args = build_parser().parse_args(["--only", "alignment", "--alignment-mode", "oblate", "--time", "90"])
    assert args.alignment_mode == "oblate"
    assert args.time == 90


def test_cluster_groups_components_and_energy():
    particles = [particle(u=[1, 0, 0]) for _ in range(25)]
    particles += [particle(u=[0, 10, 0]) for _ in range(25)]
    particles += [particle(u=[0, 0, 0])]
    table, metadata = enrich_alignment(particle_alignment(particles), particles, np.ones(51)*2)
    assert table.cluster.iloc[0] != table.cluster.iloc[25]
    assert table.cluster.iloc[0] >= 0
    assert table.cluster.iloc[-1] == -2
    assert table.kinetic_energy.iloc[25] == 100
    np.testing.assert_allclose(table.v_axis[:50]**2 + table.v_plane[:50]**2, table.speed[:50]**2)
    assert metadata["features"] == ["angle_deg", "speed"]
    np.testing.assert_allclose(metadata["feature_mean"], [45, 5.5])
    np.testing.assert_allclose(metadata["feature_scale"], [45, 4.5])


def test_cluster_noise_and_empty():
    p = [particle()]
    table, _ = enrich_alignment(particle_alignment(p), p)
    assert table.cluster.iloc[0] == -1
    empty, _ = enrich_alignment(particle_alignment([]), [])
    assert empty.empty


def test_cluster_validation():
    with pytest.raises(ValueError):
        enrich_alignment(particle_alignment([]), [], eps=0)
    with pytest.raises(ValueError, match="masses"):
        enrich_alignment(particle_alignment([particle()]), [particle()], masses=[])
