from .pattern import Pattern
from torch.utils.data import Dataset
from typing import List


class PatternDataset(Dataset):
    patterns: List[Pattern]

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx: int):
        return self.patterns[idx].get_tensor()

    def get_pattern(self, idx: int):
        return self.patterns[idx]
