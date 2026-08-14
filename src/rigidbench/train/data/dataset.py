import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Stream pre-encoded artifacts listed in a JSONL manifest (one row per sample)."""

    def __init__(self, manifest_path: str):
        self.root = Path(manifest_path).parent
        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {key: torch.load(self.root / filename, weights_only=True).squeeze(0) for key, filename in row.items()}
