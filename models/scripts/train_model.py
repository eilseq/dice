import argparse
import os
import torch
import torch.optim as optim

from dice_datasets import RandomPatternConfig
from dice_models import createDiceModel, DiceArchitecture, createDiceLoss, DiceLoss
from torch.utils.data import DataLoader


# Enable MacOSX GPU Acceleration
device = torch.device('mps')


def parse_args():
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument('--preset', type=str, required=True,
                        help='Selected pattern preset')

    parser.add_argument('--architecture', type=int, required=True,
                        help='Selected model architecture:\n 0 - Convolutional Autoencoder \n 1 - Attention UNet')

    parser.add_argument('--loss', type=int, required=True,
                        help='Selected loss function:\n 0 - Mean Square Error With Pattern Validation Penalty \n 1 - L1 Error With Pattern Validation Penalty')

    parser.add_argument('--noise_level', type=float, required=True,
                        help='Noise level used during training of diffusion model')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size used in model training')

    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs required for model training')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate of selected optimizer (Adam)')

    args = parser.parse_args()

    if args.architecture == 0:
        args.architecture = DiceArchitecture.CONVOLUTIONAL_AUTO_ENCODER
    elif args.architecture == 1:
        args.architecture = DiceArchitecture.ATTENTION_UNET

    if args.loss == 0:
        args.loss = DiceLoss.MSE_POLYPHONY_PENALTY
    elif args.loss == 1:
        args.loss = DiceLoss.L1_POLYPHONY_PENALTY

    return args


def train_model(dataset_path: str,
                dataset_config: RandomPatternConfig,
                architecture: DiceArchitecture,
                loss_function: DiceLoss,
                noise_level: float,
                num_epochs: int = 20,
                batch_size: int = 32,
                learning_rate: float = 0.001):

    data = torch.load(dataset_path)
    dataloader = DataLoader(data, batch_size=batch_size)

    model = createDiceModel(architecture).to(device)
    criterion = createDiceLoss(loss_function, config=dataset_config)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for pattern_tensor in dataloader:
            pattern_tensor = pattern_tensor.to(device)

            # required by current implementation of models, based on images with single layer (channel dimension)
            pattern_tensor = pattern_tensor.unsqueeze(1)
            print(pattern_tensor.shape)

            noise_tensor = torch.randn_like(pattern_tensor).to(device)
            noisy_pattern_tensor = pattern_tensor + noise_tensor * noise_level

            optimizer.zero_grad()
            outputs = model(noisy_pattern_tensor)
            loss = criterion(outputs, pattern_tensor)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


if __name__ == "__main__":
    args = parse_args()

    dataset_preset_path = os.path.join(
        "..", "datasets", "presets", args.preset + ".json")

    compiled_dataset_path = os.path.join(
        "..", "dist", "datasets", args.preset + ".pt")

    model = train_model(
        dataset_path=compiled_dataset_path,
        dataset_config=RandomPatternConfig.from_json(dataset_preset_path),
        architecture=args.architecture,
        loss_function=args.loss,
        noise_level=args.noise_level,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_folder = os.path.abspath(os.path.join(script_dir, "..", ".."))

    dist_folder = os.path.join(workspace_folder, "dist", "models")
    os.makedirs(dist_folder, exist_ok=True)

    # Save the trained Model
    destination_path = os.path.join(dist_folder, args.preset + ".pth")
    torch.save(model.state_dict(), destination_path)
