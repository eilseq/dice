import pytest

from dice_datasets import WeightedCluster, RandomSequenceConfig, Sequence


@pytest.fixture
def cluster_zeros():
    return WeightedCluster(triggers=[0, 0, 0, 0], weight=1.0)


@pytest.fixture
def cluster_ones():
    return WeightedCluster(triggers=[1, 1, 1, 1], weight=1.0)


@pytest.fixture
def cluster_null_weight():
    return WeightedCluster(triggers=[1, 1, 1, 1], weight=0.0)


def test_random_generation_length(cluster_zeros):
    config = RandomSequenceConfig(
        label="test",
        weighted_clusters=[cluster_zeros],
        length_in_clusters=4
    )

    sequence = Sequence.create_random(config)
    triggers_len = len(sequence.get_triggers())

    assert triggers_len == 16


def test_random_generation_with_zeros(cluster_zeros):
    config = RandomSequenceConfig(
        label="test",
        weighted_clusters=[cluster_zeros],
        length_in_clusters=4
    )

    sequence = Sequence.create_random(config)
    triggers_sum = sum(sequence.get_triggers())

    assert triggers_sum == 0


def test_random_generation_with_ones(cluster_ones):
    config = RandomSequenceConfig(
        label="test",
        weighted_clusters=[cluster_ones],
        length_in_clusters=4
    )

    sequence = Sequence.create_random(config)
    triggers_sum = sum(sequence.get_triggers())

    assert triggers_sum == 16


def test_random_generation_with_null_weight(cluster_ones, cluster_null_weight):
    config = RandomSequenceConfig(
        label="test",
        weighted_clusters=[cluster_ones, cluster_null_weight],
        length_in_clusters=4
    )

    sequence = Sequence.create_random(config)
    triggers_sum = sum(sequence.get_triggers())

    assert triggers_sum == 16


def test_random_generation_with_zeros_and_ones(cluster_zeros, cluster_ones):
    config = RandomSequenceConfig(
        label="test",
        weighted_clusters=[cluster_zeros, cluster_ones],
        length_in_clusters=4
    )

    sequence = Sequence.create_random(config)
    triggers_sum = sum(sequence.get_triggers())

    assert triggers_sum <= 16
    assert triggers_sum >= 0
