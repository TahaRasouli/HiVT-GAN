from typing import Callable, Optional
import os
import json
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from tqdm import tqdm

from datasets.nuscenes_dataset import NuScenesHiVTDataset
from utils import SimpleTokenizer

class NuScenesHiVTDataModule(LightningDataModule):
    def __init__(
        self,
        split_file: str = "balanced_splits.json", # <--- New Argument
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.split_file = split_file # Path to the JSON we created
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        self.tokenizer = SimpleTokenizer(vocab_file="vocab.json")

    def prepare_data(self) -> None:
        # Build Vocab if needed (Scan the JSON list instead of folder)
        if not os.path.exists("vocab.json"):
            print("[DataModule] Building Vocab from Split File...")
            if not os.path.exists(self.split_file):
                raise FileNotFoundError(f"Please run scripts/create_balanced_split.py first to generate {self.split_file}")
            
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
            
            # Scan training files from the list
            train_files = splits['train']
            captions = []
            
            # Scan first 2000 files for speed
            import torch
            for fpath in tqdm(train_files[:2000], desc="Scanning Vocab"):
                try:
                    data = torch.load(fpath)
                    text = data.caption_dict.get('driving_behavior', "")
                    if text: captions.append(text)
                except: pass
            
            self.tokenizer.fit(captions)
            self.tokenizer.save_vocab("vocab.json")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = NuScenesHiVTDataset(
                split_file=self.split_file,
                split="train",
                tokenizer=self.tokenizer
            )

        if stage in (None, "fit", "validate"):
            self.val_dataset = NuScenesHiVTDataset(
                split_file=self.split_file,
                split="val",
                tokenizer=self.tokenizer
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )