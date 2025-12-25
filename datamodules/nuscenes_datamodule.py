from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from datasets.nuscenes_dataset import NuScenesHiVTDataset


class NuScenesHiVTDataModule(LightningDataModule):
    """
    Lightning DataModule for HiVT-compatible nuScenes data.

    Assumes offline preprocessing has already been performed and stored as:
        root/
            train_processed/
            val_processed/
    """

    def __init__(
        self,
        root: str,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
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

    # --------------------------------------------------
    def prepare_data(self) -> None:
        """
        No-op.

        All preprocessing is assumed to be done offline.
        This method exists for Lightning compatibility.
        """
        pass

    # --------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create datasets for training and validation.
        """
        if stage in (None, "fit"):
            self.train_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="train",
                transform=self.train_transform,
                max_samples=self.max_train_samples,
            )

            self.val_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="val",
                transform=self.val_transform,
                max_samples=self.max_val_samples,
            )

    # --------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=0,              # start with 0 for stability
            pin_memory=False,           # critical
            persistent_workers=False,   # critical
        )


    # --------------------------------------------------
    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=0,              # start with 0 for stability
            pin_memory=False,           # critical
            persistent_workers=False,   # critical
        )

