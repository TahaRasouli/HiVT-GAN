import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from collections import Counter

from datasets.nuscenes_dataset import NuScenesHiVTDataset
from datasets.nuscenes_datamodule import NuScenesHiVTDataModule
from models.trajectory_generator import CVAE 
from models.maneuver_classifier import ManeuverClassifier

def calculate_6class_weights(dataset):
    print("[Info] Calculating weights for 6-class setup (ignoring U-Turns/Off-Map)...")
    counts = Counter()
    # 0:Straight, 1:Left, 2:Right, 4:LCL, 5:LCR, 6:Stat
    mapping = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5}
    
    for i in range(len(dataset)):
        data = dataset.get(i)
        m_id = int(data.maneuver_id.item())
        if m_id in mapping:
            counts[mapping[m_id]] += 1
    
    total = sum(counts.values())
    weights = torch.zeros(6)
    for idx in range(6):
        weights[idx] = total / (6 * counts[idx]) if counts[idx] > 0 else 1.0
        
    print(f"[Info] Remapped Weights: {weights}")
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

    # 1. Dataset & Weights
    train_dataset = NuScenesHiVTDataset(root=args.root, split='train')
    class_weights = calculate_6class_weights(train_dataset)
    
    # 2. DataModule
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=8
    )

    # 3. Backbone & Model
    print(f"[Info] Loading Frozen Backbone: {args.ckpt_path}")
    backbone = CVAE.load_from_checkpoint(args.ckpt_path, strict=False)
    
    model = ManeuverClassifier(
        frozen_backbone=backbone,
        num_classes=6,
        lr=args.lr,
        class_weights=class_weights
    )

    # 4. DEFINING CALLBACKS (Fixed the omission)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        filename="maneuver-{epoch:02d}-{val_f1_macro:.2f}",
        save_top_k=2,
        verbose=True
    )
    
    early_stop = EarlyStopping(
        monitor="val_f1_macro", 
        patience=10, 
        mode="max",
        verbose=True
    )

    # 5. Trainer logic
    logger = TensorBoardLogger("logs", name="maneuver_classifier")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop], # Both are now defined
        logger=logger,
        precision="16-mixed",
        log_every_n_steps=10
    )

    print("[Info] Starting Maneuver Classification Training...")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()