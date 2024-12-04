import torch.nn as nn
import torch.nn.functional as F

from dice_datasets import Pattern, RandomPatternConfig
from enum import Enum
from torch import Tensor


class DiceLoss(Enum):
    MSE_POLYPHONY_PENALTY = 1
    L1_POLYPHONY_PENALTY = 2


class MSELossWithPolyphonyRequirementsPenalty(nn.Module):
    def __init__(self, config: RandomPatternConfig):
        super(MSELossWithPolyphonyRequirementsPenalty, self).__init__()
        self.config = config

    def forward(self, input: Tensor, target: Tensor, penalty_factor: int = 10):
        mse_loss = F.mse_loss(input, target, reduction='none')
        is_valid = Pattern.tensor_valid_polyphony_requirements(
            pattern_tensor=target,
            max_polyphony=self.config.max_polyphony,
            max_num_events_with_full_polyphony=self.config.max_num_events_with_full_polyphony
        )

        if not is_valid:
            mse_loss *= penalty_factor

        return mse_loss.mean()


class L1LossWithPolyphonyRequirementsPenalty(nn.Module):
    def __init__(self, config: RandomPatternConfig):
        super(L1LossWithPolyphonyRequirementsPenalty, self).__init__()
        self.config = config

    def forward(self, input: Tensor, target: Tensor, penalty_factor: int = 10):
        mse_loss = F.l1_loss(input, target)
        is_valid = Pattern.tensor_valid_polyphony_requirements(
            pattern_tensor=target,
            max_polyphony=self.config.max_polyphony,
            max_num_events_with_full_polyphony=self.config.max_num_events_with_full_polyphony
        )

        if not is_valid:
            mse_loss *= penalty_factor

        return mse_loss.mean()
