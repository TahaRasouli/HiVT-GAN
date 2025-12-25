import os
from typing import Optional, List

import torch
from torch_geometric.data import Dataset, Batch

from utils import TemporalData


class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset.

    Assumes offline preprocessing producing .pt TemporalData files:
        root/
            train_processed/
            val_processed/
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

        self._processed_dir = os.path.join(self.root, self._directory)
        if not os.path.isdir(self._processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {self._processed_dir}")

        self._processed_file_names = sorted(
            f for f in os.listdir(self._processed_dir) if f.endswith(".pt")
        )

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    # --------------------------------------------------
    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    # --------------------------------------------------
    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    # --------------------------------------------------
    def len(self) -> int:
        return len(self._processed_file_names)

    # --------------------------------------------------
    def get(self, idx: int) -> TemporalData:
        path = os.path.join(self.processed_dir, self._processed_file_names[idx])
        data = torch.load(path)
        assert isinstance(data, TemporalData)
        return data

    # --------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)
