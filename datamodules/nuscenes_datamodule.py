from typing import Callable, Optional
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
        num_workers: int = 8,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()

        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    # ---------- SETUP CALLED ONCE ----------
    def setup(self, stage=None):

        if not hasattr(self, "train_dataset"):

            self.train_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="train",
            )

            self.val_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="val",
            )

    # ---------- TRAIN LOADER ----------
    def train_dataloader(self):

        sample_weights = []

        for data in self.train_dataset:

            label = int(data.maneuver_id.item())

            weight = 10.0 if label in [1, 2, 3, 4] else 1.0

            sample_weights.append(weight)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=NuScenesHiVTDataset.collate_fn,
        )

    # ---------- VAL LOADER ----------
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=NuScenesHiVTDataset.collate_fn,
        )
