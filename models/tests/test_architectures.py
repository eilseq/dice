import pytest
import torch

from dice_models.architectures import ConvAutoencoder, AttentionUNet


@pytest.fixture
def autoencoder():
    return ConvAutoencoder(num_channels=1)


@pytest.fixture
def unet():
    return AttentionUNet(num_channels=1)


@pytest.mark.parametrize("input_shape", [
    (1, 1, 8, 8),    # Smaller size
    (1, 1, 16, 16),  # Square input
    (1, 1, 32, 32),  # Larger square
    (1, 1, 64, 64),  # Larger size
    (1, 1, 16, 32),  # Rectangular input
    (1, 1, 32, 16),  # Transposed rectangle
    (32, 1, 8, 8),   # Batched Tensor
])
def test_autoencoder_output_shapes(autoencoder, input_shape):
    tensor = torch.rand(*input_shape, dtype=torch.float32)
    output = autoencoder(tensor)
    assert output.shape == tensor.shape


@pytest.mark.parametrize("input_shape", [
    (1, 1, 8, 8),    # Smaller size
    (1, 1, 16, 16),  # Square input
    (1, 1, 32, 32),  # Larger square
    (1, 1, 64, 64),  # Larger size
    (1, 1, 16, 32),  # Rectangular input
    (1, 1, 32, 16),  # Transposed rectangle
    (32, 1, 8, 8),   # Batched Tensor
])
def test_unet_output_shapes(unet, input_shape):
    tensor = torch.rand(*input_shape, dtype=torch.float32)
    output = unet(tensor)
    assert output.shape == tensor.shape
