import random
import torch

from dataclasses import dataclass
from itertools import chain
from typing import List


# Enable MacOSX GPU Acceleration
device = torch.device('mps')


@dataclass
class Cluster:
    triggers: List[int]


@dataclass
class WeightedCluster(Cluster):
    weight: int = 1

    @staticmethod
    def from_dictionary(dict: dict):
        return WeightedCluster(**dict)


@dataclass
class RandomSequenceConfig:
    label: str
    weighted_clusters: List['WeightedCluster']
    length_in_clusters: int = 4

    @staticmethod
    def from_dictionary(dict: dict):
        weighted_clusters = [
            WeightedCluster.from_dictionary(cluster_dict) for cluster_dict in dict['weighted_clusters']
        ]

        return RandomSequenceConfig(
            label=dict['label'],
            weighted_clusters=weighted_clusters,
            length_in_clusters=dict['length_in_clusters']
        )


class Sequence:
    def __init__(self, label: str, clusters: List[Cluster]):
        self.label = label
        self.clusters = clusters

    def get_triggers(self):
        # Returns the triggers of sequence into one list
        triggers = list(
            chain(*[cluster.triggers for cluster in self.clusters]))
        return triggers

    def get_tensor(self):
        # Returns the triggers of sequence into a 1D tensor
        tensor = torch.tensor(
            self.get_triggers(), device=device, dtype=torch.float32).view(-1)
        return tensor

    @staticmethod
    def create_random(config: RandomSequenceConfig) -> 'Sequence':
        """
        Generates a Sequence from a RandomSequenceMap. It selects clusters based on their
        weighted probability, then creates the sequence by flattening the triggers of selected clusters.
        """
        weights = [cluster.weight for cluster in config.weighted_clusters]

        # Choose clusters based on weighted probabilities
        chosen_clusters = random.choices(
            config.weighted_clusters, weights=weights, k=config.length_in_clusters)

        return Sequence(config.label, chosen_clusters)
