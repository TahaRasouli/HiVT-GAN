import os
from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

# Import modules
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.cvae import CVAE
from models.maneuver_classifier import ManeuverClassifier
from datasets.nuscenes_dataset import NuScenesHiVTDataset

# Optimization
torch.set_float32_matmul_precision('medium')

def calculate_class_weights(dataset):
    """
    Scans the dataset to compute inverse frequency weights.
    Prevents the model from ignoring U-Turns and only predicting Straight.
    """
    print(f"\n[Info] Scanning {len(dataset)} samples to calculate Class Weights...")
    
    counts = {i: 0 for i in range(7)}
    total = 0
    
    # Iterate with progress bar
    for i in tqdm(range(len(dataset)), desc="Computing Weights"):
        try:
            data = dataset.get(i)
            # Handle tensor vs int
            if isinstance(data.maneuver_id, torch.Tensor):
                mid = data.maneuver_id.item()
            else:
                mid = int(data.maneuver_id)
            
            if 0 <= mid < 7:
                counts[mid] += 1
                total += 1
        except Exception as e:
            # Skip corrupted samples if any
            continue

    print(f"[Info] Class Counts: {counts}")

    # Formula: W_c = N_total / (N_classes * Count_c)
    weights = []
    n_classes = 7
    for i in range(n_classes):
        c = counts.get(i, 0)
        if c > 0:
            weights.append(total / (n_classes * c))
        else:
            # High penalty if class is missing/rare to encourage learning it later
            weights.append(2.0) 
    
    # Convert to tensor
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"[Info] Final Calculated Weights: {weight_tensor}\n")
    return weight_tensor

def main():
    pl.seed_everything(2024)
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--root", type=str, required=True, help="Path to processed data folder")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to frozen HiVT/CVAE .ckpt file")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Calculate Weights (Handling Imbalance)
    # ---------------------------------------------------------
    # We load the dataset directly first to scan statistics
    train_dataset = NuScenesHiVTDataset(root=args.root, split='train')
    class_weights = calculate_class_weights(train_dataset)
    
    # ---------------------------------------------------------
    # 2. Setup DataModule
    # ---------------------------------------------------------
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # 3. Load Backbone & Initialize Classifier
    # ---------------------------------------------------------
    print(f"[Info] Loading Frozen Backbone from: {args.ckpt_path}")
    
    # Load CVAE (strict=False ignores extra GAN keys if present)
    backbone = CVAE.load_from_checkpoint(args.ckpt_path)
    
    # Initialize our wrapping model
    model = ManeuverClassifier(
        frozen_backbone=backbone,
        num_classes=7,
        learning_rate=args.lr,
        class_weights=class_weights
    )

    # ---------------------------------------------------------
    # 4. Trainer
    # ---------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="caption-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2
    )
    
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=50,
        strategy=DDPStrategy(find_unused_parameters=False),
        check_val_every_n_epoch=1 # Ensure validation runs every epoch to see prints
    )

    print("[Info] Starting Linear Probe Training...")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()