import os
from typing import Optional, List

import torch
from torch_geometric.data import Dataset, Batch

from utils import TemporalData


class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset.
    Assumes offline preprocessing to .pt TemporalData files.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self._directory = f"{split}_processed"
        self.root = root
        self.transform = transform

        # IMPORTANT: use a private attribute (PyG already has processed_dir)
        self._processed_dir = os.path.join(self.root, self._directory)
        if not os.path.isdir(self._processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {self._processed_dir}")

        self.processed_paths = sorted(
            os.path.join(self._processed_dir, f)
            for f in os.listdir(self._processed_dir)
            if f.endswith(".pt")
        )

        if max_samples is not None:
            self.processed_paths = self.processed_paths[:max_samples]

        super().__init__(root, transform=transform)

    # --------------------------------------------------
    def len(self) -> int:
        return len(self.processed_paths)

    # --------------------------------------------------
    def get(self, idx: int) -> TemporalData:
        data = torch.load(self.processed_paths[idx])
        assert isinstance(data, TemporalData)
        return data

    # --------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)
