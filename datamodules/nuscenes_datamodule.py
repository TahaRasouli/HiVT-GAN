import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from datasets.nuscenes_dataset import NuScenesHiVTDataset

class NuScenesHiVTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        split_file: str,
        train_batch_size: int,
        val_batch_size: int,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        tokenizer=None,  # <--- Accept the BERT tokenizer here
        **kwargs
    ):
        super(NuScenesHiVTDataModule, self).__init__()
        self.root = root
        self.split_file = split_file
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.tokenizer = tokenizer # Store it

    def prepare_data(self):
        # Optional: any downloading logic
        pass

    def setup(self, stage=None):
        # Pass the tokenizer to the datasets
        self.train_dataset = NuScenesHiVTDataset(
            split_file=self.split_file,
            split="train",
            root=self.root,
            tokenizer=self.tokenizer 
        )
        
        self.val_dataset = NuScenesHiVTDataset(
            split_file=self.split_file,
            split="val",
            root=self.root,
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
            collate_fn=NuScenesHiVTDataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=NuScenesHiVTDataset.collate_fn
        )