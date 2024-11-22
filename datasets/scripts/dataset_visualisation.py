import argparse
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dice_datasets import PatternDataset
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Selected pattern dataset')
    args = parser.parse_args()
    return args.dataset


def create_pattern_visualization_with_slider(dataset: PatternDataset):
    ax_pattern = plt.axes([0.15, 0.15, 0.7, 0.8])

    def update_visualization(index):
        index = int(index)
        pattern = dataset.get_pattern(index)
        df = pd.DataFrame(pattern.get_triggers())
        df.index = pattern.get_labels()
        sns.heatmap(df, cmap="GnBu", cbar=False, ax=ax_pattern)

    # Initialize the first visualization
    update_visualization(0)

    # Add a slider for selecting patterns
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03],
                         facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Pattern', 0, len(
        dataset) - 1, valinit=0, valstep=1)

    # Update the visualization whenever the slider is adjusted
    slider.on_changed(update_visualization)

    plt.show()


if __name__ == "__main__":
    dataset_name = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_folder = os.path.abspath(os.path.join(script_dir, "..", ".."))

    dist_folder = os.path.join(workspace_folder, "dist", "datasets")
    os.makedirs(dist_folder, exist_ok=True)

    source_path = os.path.join(dist_folder, dataset_name + ".pt")
    dataset: PatternDataset = torch.load(source_path, weights_only=False)

    create_pattern_visualization_with_slider(dataset)
