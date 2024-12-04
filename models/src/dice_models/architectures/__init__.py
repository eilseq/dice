from .autoencoder import ConvAutoencoder
from .unet import AttentionUNet
from enum import Enum


class DiceArchitecture(Enum):
    CONVOLUTIONAL_AUTO_ENCODER: 1
    ATTENTION_UNET: 2


__all__ = ["DiceArchitecture", "AttentionUNet", "ConvAutoencoder"]
