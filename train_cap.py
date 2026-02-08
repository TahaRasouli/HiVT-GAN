import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from collections import Counter

from datasets.nuscenes_dataset import NuScenesHiVTDataset
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.cvae import CVAE
from models.maneuver_classifier import ManeuverClassifier

def calculate_6class_weights(dataset):
    print("[Info] Calculating inverse-frequency weights for 6 classes (excluding U-Turns)...")
    counts = Counter()
    for i in range(len(dataset)):
        data = dataset.get(i)
        m_id = int(data.maneuver_id.item())
        if m_id == 3:  # Skip U-Turns (ID 3)
            continue
        counts[m_id] += 1

    total = sum(counts.values())
    weights = torch.zeros(6)  # 6 classes: 0,1,2,4,5,6
    for cls in range(7):
        if cls == 3:
            continue
        weights[cls - (1 if cls > 3 else 0)] = total / (6 * counts[cls]) if counts[cls] > 0 else 1.0
    return weights
    counts = Counter()
    for i in range(len(dataset)):
        # Quick access to label without full sanitization if possible
        data = dataset.get(i)
        counts[int(data.maneuver_id.item())] += 1
    
    total = sum(counts.values())
    weights = torch.zeros(7)
    for cls in range(7):
        # Inverse frequency: total / (num_classes * count_per_class)
        weights[cls] = total / (7 * counts[cls]) if counts[cls] > 0 else 1.0
    return weights

def main():
    pl.seed_everything(2024)
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--devices", type=int, default=1)
    
    args = parser.parse_args()

    # 1. Initialize Dataset & Calculate Weights
    # This will trigger the filtering progress bar we added to nuscenes_dataset.py
    train_dataset = NuScenesHiVTDataset(root=args.root, split='train')
    class_weights = calculate_6class_weights(train_dataset)
    
    # 2. Setup DataModule
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=8
    )

    # 3. Load Frozen Backbone & Model
    print(f"[Info] Loading Frozen Backbone: {args.ckpt_path}")
    backbone = CVAE.load_from_checkpoint(args.ckpt_path, strict=False)
    
    model = ManeuverClassifier(
        frozen_backbone=backbone,
        num_classes=6,
        lr=args.lr,
        class_weights=class_weights
    )

    # 4. Define Callbacks (Fixes the NameError)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        filename="maneuver-classifier-{epoch:02d}-{val_f1_macro:.2f}",
        save_top_k=2
    )
    
    early_stop = EarlyStopping(
        monitor="val_f1_macro", 
        patience=10, 
        mode="max"
    )

    # 5. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        precision="16-mixed",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10
    )

    print("[Info] Starting Maneuver Classification Training...")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()