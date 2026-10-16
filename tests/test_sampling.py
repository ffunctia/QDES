import pytest

from qdes.simulator import QuantumSystem


def test_independent_groups_sample_joint_distribution_reproducibly():
    system = QuantumSystem(2)
    system.apply("H", [0])
    system.apply("H", [1])

    first = system.sample(count=8192, seed=12345)
    repeated = system.sample(count=8192, seed=12345)

    assert first == repeated
    assert set(first) == {"00", "01", "10", "11"}
    for count in first.values():
        assert count / 8192 == pytest.approx(0.25, abs=0.025)
