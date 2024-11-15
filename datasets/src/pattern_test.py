import pytest

from pattern import Pattern, RandomPatternConfig
from sequence import RandomSequenceConfig, WeightedCluster


@pytest.fixture
def ones_and_zeros_random_pattern_config():
    ones_and_zeros = RandomSequenceConfig(
        label="ones_and_zeros",
        length_in_clusters=4,
        weighted_clusters=[
            WeightedCluster(
                triggers=[1, 1, 1, 1],
                weight=1
            ),
            WeightedCluster(
                triggers=[0, 0, 0, 0],
                weight=1
            )
        ]
    )

    return RandomPatternConfig(
        max_polyphony=2,
        random_sequence_configs=[
            ones_and_zeros, ones_and_zeros, ones_and_zeros, ones_and_zeros
        ]
    )


def test_random_pattern_polyphony(ones_and_zeros_random_pattern_config):
    pattern = Pattern.create_random(ones_and_zeros_random_pattern_config)
    assert pattern.meet_polyphony_requirements(
        ones_and_zeros_random_pattern_config.max_polyphony) == True
