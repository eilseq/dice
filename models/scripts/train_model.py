import argparse
import json
import os
import torch
import torch.optim as optim

from dice_datasets import RandomPatternConfig
from dice_models import createDiceModel, DiceArchitecture, createDiceLoss, DiceLoss
from rich.progress import Progress, Live, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from torch.utils.data import DataLoader


# Enable MacOSX GPU Acceleration
device = torch.device('mps')


def parse_args():
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument('--id', type=str, required=True,
                        help='Model identifier')

    parser.add_argument('--preset', type=str, required=True,
                        help='Selected pattern preset')

    parser.add_argument('--architecture', type=str, required=True,
                        help='Selected model architecture:\n conv_auto_enc - Convolutional Autoencoder \n att_unet - Attention UNet')

    parser.add_argument('--loss', type=str, required=True,
                        help='Selected loss function:\n mse_poly_penalty - Mean Square Error With Pattern Validation Penalty \n l1_poly_penalty - L1 Error With Pattern Validation Penalty')

    parser.add_argument('--noise_level', type=float, required=True,
                        help='Noise level used during training of diffusion model, usually between 0.1 and 0.2')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used in model training')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs required for model training')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of selected optimizer (Adam)')

    return parser.parse_args()


def train_model(dataset_path: str,
                dataset_config: RandomPatternConfig,
                architecture: DiceArchitecture,
                loss_function: DiceLoss,
                noise_level: float,
                num_epochs: int = 20,
                batch_size: int = 32,
                learning_rate: float = 0.001):

    data = torch.load(dataset_path, weights_only=False)
    dataloader = DataLoader(data, batch_size=batch_size)

    model = createDiceModel(architecture).to(device)
    criterion = createDiceLoss(loss_function, config=dataset_config)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TextColumn(
            "[cyan]Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}"),
        TextColumn(
            "[green]Entries: {task.fields[entries_used]}/{task.fields[total_entries]}"),
        TextColumn("[red]Loss: {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
    )

    # Training loop
    with Live(progress, refresh_per_second=10):
        task_id = progress.add_task(
            "[cyan]Epoch progress...",
            total=len(data),
            total_entries=len(data),
            entries_used=0,
            total_epochs=num_epochs,
            epoch=0,
            loss=1.0
        )

        for epoch in range(num_epochs):
            progress.update(task_id, epoch=epoch)

            for i, pattern_tensor in enumerate(dataloader, start=1):
                pattern_tensor = pattern_tensor.to(device)

                # required by model's - multi-channel architecture
                pattern_tensor = pattern_tensor.unsqueeze(1)

                noise_tensor = torch.randn_like(pattern_tensor).to(device)
                noisy_pattern_tensor = pattern_tensor + noise_tensor * noise_level

                optimizer.zero_grad()
                outputs = model(noisy_pattern_tensor)
                loss = criterion(outputs, pattern_tensor)

                loss.backward()
                optimizer.step()

                # Update progress
                progress.update(
                    task_id,
                    advance=1,
                    loss=loss.item(),
                    entries_used=i
                )

    return model


if __name__ == "__main__":
    args = parse_args()

    # Compose Import Paths
    dataset_preset_path = os.path.join(
        "..", "datasets", "presets", args.preset + ".json")

    compiled_dataset_path = os.path.join(
        "..", "dist", "datasets", args.preset + ".pt")

    # Set Distribution Folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_folder = os.path.abspath(os.path.join(script_dir, "..", ".."))

    dist_folder = os.path.join(workspace_folder, "dist", "models")
    os.makedirs(dist_folder, exist_ok=True)

    # Train Model and Store Result
    model = train_model(
        dataset_path=compiled_dataset_path,
        dataset_config=RandomPatternConfig.from_json(dataset_preset_path),
        architecture=args.architecture,
        loss_function=args.loss,
        noise_level=args.noise_level,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate)

    # Store Model and Config data
    destination_path = os.path.join(
        dist_folder, args.preset + "-" + args.architecture + "-" + args.loss
    )

    torch.save(model.state_dict(), destination_path + ".pth")

    with open(destination_path + ".json", "w") as outfile:
        outfile.write(json.dumps(vars(args)))
