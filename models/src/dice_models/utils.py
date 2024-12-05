from .architectures import DiceArchitecture, ConvAutoencoder, AttentionUNet
from .loss_functions import DiceLoss, MSELossWithPolyphonyRequirementsPenalty, L1LossWithPolyphonyRequirementsPenalty

from dice_datasets import RandomPatternConfig


def createDiceModel(architecture: DiceArchitecture, num_channels: int = 1):
    match DiceArchitecture(architecture):
        case DiceArchitecture.CONVOLUTIONAL_AUTO_ENCODER:
            return ConvAutoencoder(num_channels)
        case DiceArchitecture.ATTENTION_UNET:
            return AttentionUNet(num_channels)


def createDiceLoss(loss_function: DiceLoss, config: RandomPatternConfig):
    match DiceLoss(loss_function):
        case DiceLoss.MSE_POLYPHONY_PENALTY:
            return MSELossWithPolyphonyRequirementsPenalty(config)
        case DiceLoss.L1_POLYPHONY_PENALTY:
            return L1LossWithPolyphonyRequirementsPenalty(config)
