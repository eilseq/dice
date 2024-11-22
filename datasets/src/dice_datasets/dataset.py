from .pattern import Pattern
from torch.utils.data import Dataset
from typing import List


class PatternDataset(Dataset):
    def __init__(self, patterns: List[Pattern]):
        self.pattern_tensors = [
            pattern.get_tensor() for pattern in patterns
        ]

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        return self.pattern_tensors[idx]
