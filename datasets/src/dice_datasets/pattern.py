import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from dataclasses import dataclass
from .sequence import RandomSequenceConfig, Sequence
from typing import List


@dataclass
class RandomPatternConfig:
    random_sequence_configs: List[RandomSequenceConfig]
    max_polyphony: int
    max_num_events_with_full_polyphony: int

    @staticmethod
    def from_dictionary(dict: dict):
        random_sequence_configs = [
            RandomSequenceConfig.from_dictionary(seq_config) for seq_config in dict['random_sequence_configs']
        ]

        # for readability purposes JSON has inverted order
        random_sequence_configs.reverse()

        return RandomPatternConfig(
            max_polyphony=dict['max_polyphony'],
            max_num_events_with_full_polyphony=dict['max_num_events_with_full_polyphony'],
            random_sequence_configs=random_sequence_configs
        )

    @staticmethod
    def from_json(path: str):
        with open(path, 'r') as file:
            dict_config = json.load(file)
            return RandomPatternConfig.from_dictionary(dict_config)


@dataclass
class Pattern:
    sequences: List[Sequence]

    def get_triggers(self):
        # Returns the triggers of pattern into one matrix
        return [sequence.get_triggers() for sequence in self.sequences]

    def get_tensor(self):
        # Returns the triggers of sequence into a 2D tensor
        tensors = [sequence.get_tensor() for sequence in self.sequences]
        return torch.stack(tensors, dim=0)

    def meet_polyphony_requirements(self, max_polyphony: int, max_num_events_with_full_polyphony: int):
        polyphony = torch.sum(self.get_tensor(), dim=0)

        # polyphony should be limited to a defined maximum
        if torch.max(polyphony) > max_polyphony:
            return False

        # and said maximum should be reached for a limited number ot times
        if torch.sum(torch.floor(polyphony/max_polyphony)) > max_num_events_with_full_polyphony:
            return False

        return True

    def visualize(self):
        df = pd.DataFrame(self.triggers)
        plt.figure(figsize=(9, 6))
        sns.heatmap(df.T, cmap="GnBu", cbar=False)
        plt.show()

    @staticmethod
    def create_random(config: RandomPatternConfig):
        pattern = Pattern._create_random_candidate(config)

        while not pattern.meet_polyphony_requirements(config.max_polyphony, config.max_num_events_with_full_polyphony):
            pattern = Pattern._create_random_candidate(config)

        return pattern

    @staticmethod
    def _create_random_candidate(config: RandomPatternConfig):
        return Pattern([
            Sequence.create_random(sequence_config) for sequence_config in config.random_sequence_configs
        ])
