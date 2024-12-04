import pytest
import torch

from dice_datasets import Pattern, RandomPatternConfig, RandomSequenceConfig, WeightedCluster
from dice_models.loss_functions import MSELossWithPolyphonyRequirementsPenalty, L1LossWithPolyphonyRequirementsPenalty


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
        max_num_events_with_full_polyphony=2,
        random_sequence_configs=[
            ones_and_zeros, ones_and_zeros, ones_and_zeros, ones_and_zeros
        ]
    )


def test_valid_target_with_no_penalty(ones_and_zeros_random_pattern_config):
    pattern = Pattern.create_random(ones_and_zeros_random_pattern_config)
    tensor = pattern.get_tensor()

    mse_loss = MSELossWithPolyphonyRequirementsPenalty(
        config=ones_and_zeros_random_pattern_config
    )
    loss = mse_loss.forward(tensor, tensor, penalty_factor=10)

    assert loss == 0


def test_invalid_target_with_penalty(ones_and_zeros_random_pattern_config):
    tensor = torch.ones(1, 1, 16, 16, dtype=torch.float32)

    mse_loss = MSELossWithPolyphonyRequirementsPenalty(
        config=ones_and_zeros_random_pattern_config
    )
    loss = mse_loss.forward(tensor, tensor, penalty_factor=10)

    assert loss == 0
