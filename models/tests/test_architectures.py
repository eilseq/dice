import pytest
import torch

from dice_models.architectures import ConvAutoencoder, AttentionUNet


@pytest.fixture
def autoencoder():
    return ConvAutoencoder(num_channels=1)


@pytest.fixture
def unet():
    return AttentionUNet(num_channels=1)


def test_autoencoder_output_shape(autoencoder):
    tensor = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    output = autoencoder(tensor)
    assert output.shape == tensor.shape


def test_unet_output_shape(unet):
    tensor = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    output = unet(tensor)
    assert output.shape == tensor.shape
