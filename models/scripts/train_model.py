import argparse
import json
import os
import torch
import torch.optim as optim

from dice_datasets import Pattern, RandomPatternConfig
from dice_models import createDiceModel, createDiceLoss
from rich.progress import Progress, Live, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.console import Console
from rich.text import Text
from torch.utils.data import DataLoader, random_split

# Enable MacOSX GPU Acceleration
device = torch.device('mps')

console = Console()
progress = Progress(
    "[progress.description]{task.description}",
    TextColumn(
        "[green]Entries: {task.fields[entries_used]}/{task.fields[total_entries]}"),
    TextColumn("[red]Loss: {task.fields[loss]:.4f}"),
    TextColumn("[purple]Validation Loss: {task.fields[validation_loss]:.4f}"),
    TimeElapsedColumn(),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DICE model")
    parser.add_argument('--id', type=str, required=True,
                        help='Model identifier')
    parser.add_argument('--preset', type=str,
                        required=True, help='Pattern preset')
    parser.add_argument('--architecture', type=str,
                        required=True, help='Model architecture')
    parser.add_argument('--loss', type=str, required=True,
                        help='Loss function')
    parser.add_argument('--noise_level', type=float,
                        default=0.5, help='Noise level for training')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    return parser.parse_args()


def prepare_data(dataset_path, split_ratios=(0.75, 0.1, 0.15)):
    data = torch.load(dataset_path, weights_only=False)
    train_size = int(split_ratios[0] * len(data))
    val_size = int(split_ratios[1] * len(data))
    test_size = len(data) - train_size - val_size

    train_data, val_data, test_data = random_split(
        data, [train_size, val_size, test_size])

    return train_data, val_data, test_data


def train(epochs, dataset_config, dataloaders, model, criterion, optimizer, noise_level):
    train_loader, val_loader, test_loader = dataloaders

    with Live(progress, refresh_per_second=10):
        for epoch in range(epochs):
            task_id = progress.add_task(
                f"[cyan]Epoch {epoch + 1}/{epochs} -",
                total=len(train_loader),
                total_entries=len(train_loader)*train_loader.batch_size,
                entries_used=0,
                loss=1.0,
                validation_loss=1.0,
            )

            def add_channel_dimension(pattern_tensor):
                return pattern_tensor.unsqueeze(1)

            def add_noise(pattern_tensor):
                noise_tensor = torch.randn_like(pattern_tensor)
                return pattern_tensor + noise_tensor * noise_level

            # Training Loop
            model.train()
            for i, pattern_tensor in enumerate(train_loader, start=1):
                pattern_tensor = add_channel_dimension(pattern_tensor)
                optimizer.zero_grad()
                outputs = model(add_noise(pattern_tensor))
                loss = criterion(outputs, pattern_tensor)
                loss.backward()
                optimizer.step()
                progress.update(task_id, entries_used=i *
                                train_loader.batch_size, loss=loss)

            # Validation Loop
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for pattern_tensor in val_loader:
                    pattern_tensor = add_channel_dimension(pattern_tensor)
                    outputs = model(add_noise(pattern_tensor))
                    validation_loss += criterion(outputs,
                                                 pattern_tensor) / len(val_loader)
                    progress.update(task_id, validation_loss=validation_loss)

            progress.stop_task(task_id)  # Stop timer for the current epoch

    model.eval()
    accuracy = 0
    with torch.no_grad():
        for pattern_tensor in test_loader:
            pattern_tensor = add_channel_dimension(pattern_tensor)
            outputs = model(add_noise(pattern_tensor))
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            outputs = outputs.squeeze(1)

            for output in outputs:
                if Pattern.tensor_valid_polyphony_requirements(
                        output, dataset_config.max_polyphony, dataset_config.max_num_events_with_full_polyphony):
                    accuracy += 1 / (test_loader.batch_size * len(test_loader))
    return accuracy


if __name__ == "__main__":
    console.print("[bold cyan]DICE Model Training")
    args = parse_args()

    workspace_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))

    dist_folder = os.path.join(workspace_folder, "dist")

    presets_folder = os.path.join(workspace_folder, "datasets", "presets")
    datasets_folder = os.path.join(dist_folder, "datasets")
    models_folder = os.path.join(dist_folder, "models")

    os.makedirs(models_folder, exist_ok=True)
    compiled_dataset_path = os.path.join(
        "..", "dist", "datasets", args.preset + ".pt")

    dataset_config_path = os.path.join(presets_folder,  args.preset + ".json")
    dataset_config = RandomPatternConfig.from_json(dataset_config_path)

    compiled_dataset_path = os.path.join(datasets_folder, args.preset + ".pt")
    train_data, val_data, test_data = prepare_data(compiled_dataset_path)

    model = createDiceModel(args.architecture).to(device)
    accuracy = train(
        epochs=args.epochs,
        dataset_config=dataset_config,
        dataloaders=[
            DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_data, batch_size=args.batch_size, shuffle=False),
            DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        ],
        model=model,
        criterion=createDiceLoss(args.loss, config=dataset_config),
        optimizer=optim.Adam(model.parameters(), lr=args.learning_rate),
        noise_level=args.noise_level
    )

    console.print(Text(f"Final Accuracy: {accuracy:.2f}", style="bold green"))

    model_filename = f"{args.preset}-{args.architecture}-{args.loss}"
    destination_path = os.path.join(models_folder, model_filename)

    torch.save(model.state_dict(), destination_path + ".pth")
    with open(destination_path + ".json", "w") as outfile:
        json.dump(vars(args), outfile)

    console.print(Text(f"Saved at destination {
                  destination_path + ".pth"}", style="yellow"))
