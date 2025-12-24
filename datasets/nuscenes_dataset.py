import os
from typing import Optional, List

import torch
from torch_geometric.data import Dataset, Batch

from utils import TemporalData


class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset.

    This dataset assumes that all preprocessing has already been done
    offline and stored as TemporalData (.pt files), one file per sample.

    Directory structure:
        root/
            train_processed/
                <sample_token>.pt
            val_processed/
                <sample_token>.pt
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root: Root directory containing <split>_processed/
            split: "train" or "val"
            transform: optional PyG transform
            max_samples: optional limit for debugging
        """
        self.split = split
        self._directory = f"{split}_processed"
        self.root = root
        self.transform = transform

        self.processed_dir = os.path.join(self.root, self._directory)
        if not os.path.isdir(self.processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {self.processed_dir}")

        self.processed_paths = sorted(
            os.path.join(self.processed_dir, f)
            for f in os.listdir(self.processed_dir)
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
        """
        Collate TemporalData objects into a PyG Batch.
        """
        return Batch.from_data_list(batch)
