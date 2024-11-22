import argparse
import os
import torch

from dice_datasets import PatternDataset, Pattern, RandomPatternConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from rich.progress import Progress, Live, BarColumn, TimeElapsedColumn, TaskProgressColumn


def parse_args():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--preset', type=str, required=True,
                        help='Selected pattern preset')

    parser.add_argument('--size', type=str, required=True,
                        help='Selected pattern preset')

    args = parser.parse_args()
    return args.preset, int(args.size)


def create_random_dataset(preset_path, num_patterns):
    patterns = Queue()
    config = RandomPatternConfig.from_json(preset_path)

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )

    with Live(progress, refresh_per_second=10):
        task_id = progress.add_task(
            "[cyan]Generating Patterns...", total=num_patterns)

        def create_random_pattern():
            pattern = Pattern.create_random(config)
            progress.update(task_id, advance=1)
            return pattern

        with ThreadPoolExecutor(10) as executor:
            futures = [
                executor.submit(create_random_pattern) for _ in range(num_patterns)
            ]

        for future in as_completed(futures):
            pattern = future.result()
            patterns.put(pattern)

    return PatternDataset(list(patterns.queue))


if __name__ == "__main__":
    preset, num_patterns = parse_args()

    preset_path = os.path.join("presets", preset + ".json")
    dataset = create_random_dataset(preset_path, num_patterns)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_folder = os.path.abspath(os.path.join(script_dir, "..", ".."))

    dist_folder = os.path.join(workspace_folder, "dist", "datasets")
    os.makedirs(dist_folder, exist_ok=True)

    destination_path = os.path.join(dist_folder, preset + ".pt")
    torch.save(dataset, destination_path)
