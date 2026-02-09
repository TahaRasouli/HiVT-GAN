# datamodules/nuscenes_datamodule.py
from typing import Callable, Optional
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

from datasets.nuscenes_dataset import NuScenesHiVTDataset

class NuScenesHiVTDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="train",
                transform=self.train_transform,
                max_samples=self.max_train_samples,
            )
        if stage in (None, "fit", "validate"):
            self.val_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="val",
                transform=self.val_transform,
                max_samples=self.max_val_samples,
            )

    def train_dataloader(self):
        # Weighted sampler based on maneuver_id
        sample_weights = []
        for data in self.train_dataset:
            label = int(data.maneuver_id.item())
            # Weight active classes more (optional)
            weight = 10.0 if label in [1, 2, 4, 5] else 1.0
            sample_weights.append(weight)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            shuffle=False,  # sampler handles shuffling
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.val_dataset.collate_fn
        )
