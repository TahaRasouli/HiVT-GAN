from typing import Callable, Optional
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from torch.utils.data import WeightedRandomSampler  # <--- NEW IMPORT

from datasets.nuscenes_dataset import NuScenesHiVTDataset

class NuScenesHiVTDataModule(LightningDataModule):
    """
    Lightning DataModule for HiVT-compatible nuScenes data.
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
        pass

    # --------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        # Create training dataset only during 'fit'
        if stage in (None, "fit"):
            self.train_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="train",
                transform=self.train_transform,
                max_samples=self.max_train_samples,
            )

        # Create validation dataset during 'fit' OR 'validate'
        if stage in (None, "fit", "validate"):
            self.val_dataset = NuScenesHiVTDataset(
                root=self.root,
                split="val",
                transform=self.val_transform,
                max_samples=self.max_val_samples,
            )

    # --------------------------------------------------
    #  THIS IS THE CRITICAL SECTION WE MODIFIED
    # --------------------------------------------------
    def train_dataloader(self):
        # 1. SCAN DATASET FOR WEIGHTS
        # We assume the dataset is already loaded in memory or allows fast iteration
        print(f"[Info] Scanning {len(self.train_dataset)} samples to calculate Sampling Weights...")
        
        sample_weights = []
        for data in self.train_dataset:
            # Handle both Tensor and Int types safely
            if hasattr(data.maneuver_id, 'item'):
                label = int(data.maneuver_id.item())
            else:
                label = int(data.maneuver_id)
            
            # --- WEIGHT ASSIGNMENT LOGIC ---
            if label == 3:       # U-TURN (The rarest class)
                weight = 100.0   # Massive weight to force it into batches
            elif label in [1, 2, 4, 5]: # TURNS & LANE CHANGES
                weight = 10.0    # Moderate weight
            else:                # STRAIGHT (0) & STATIONARY (6)
                weight = 1.0     # Low weight (they are abundant)
            
            sample_weights.append(weight)

        # 2. CREATE SAMPLER
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )

        # 3. RETURN DATALOADER
        # Note: shuffle must be False when sampler is used
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,      # <--- The Magic Component
            shuffle=False,        # MUST be False
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    # --------------------------------------------------
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )