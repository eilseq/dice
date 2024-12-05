from .autoencoder import ConvAutoencoder
from .unet import AttentionUNet
from enum import Enum


class DiceArchitecture(Enum):
    CONVOLUTIONAL_AUTO_ENCODER = "conv_auto_enc"
    ATTENTION_UNET = "att_unet"


__all__ = ["DiceArchitecture", "AttentionUNet", "ConvAutoencoder"]
