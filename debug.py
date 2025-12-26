import torch
from pytorch_lightning import Trainer

from models.hivt import HiVT
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# --------------------------------------------------
# CONFIG — KEEP THIS MINIMAL
# --------------------------------------------------
ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes-hivt/"

TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1

LIMIT_TRAIN_BATCHES = 5
LIMIT_VAL_BATCHES = 5

# --------------------------------------------------
# DataModule (MATCHES YOUR IMPLEMENTATION)
# --------------------------------------------------
datamodule = NuScenesHiVTDataModule(
    root=ROOT,
    train_batch_size=TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,

    # IMPORTANT: keep everything conservative
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,

    # Optional: hard limit samples (extra safety)
    max_train_samples=10,
    max_val_samples=10,
)

datamodule.setup("fit")

# --------------------------------------------------
# Model
# --------------------------------------------------
model = HiVT(
    historical_steps=20,
    future_steps=30,
    num_modes=6,
    rotate=True,
    node_dim=2,
    edge_dim=2,
    embed_dim=128,
    num_heads=8,
    dropout=0.1,
    num_temporal_layers=4,
    num_global_layers=3,
    local_radius=50,
    parallel=False,
    lr=5e-4,
    weight_decay=1e-4,
    T_max=64,

    # GAN ENABLED (this is what we are testing)
    use_gan=True,
    lambda_adv=0.05,
    critic_steps=1,
)

# ... (DataModule and Model setup remains the same)

# --------------------------------------------------
# PRE-FLIGHT DATA CHECK
# --------------------------------------------------
train_loader = datamodule.train_dataloader()
sample_batch = next(iter(train_loader))
print(f"Checking first batch... y has NaNs: {torch.isnan(sample_batch.y).any()}")
print(f"Checking first batch... x has NaNs: {torch.isnan(sample_batch.x).any()}")

# --------------------------------------------------
# Trainer — ENHANCED DEBUG MODE
# --------------------------------------------------
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=1,
    limit_train_batches=LIMIT_TRAIN_BATCHES,
    limit_val_batches=LIMIT_VAL_BATCHES,
    log_every_n_steps=1,
    enable_checkpointing=False,
    # Use 32-bit precision to rule out underflow/overflow issues
    precision=32, 
)

# --------------------------------------------------
# RUN WITH ANOMALY DETECTION
# --------------------------------------------------
print("\n🚀 Running HiVT-GAN sanity check with Anomaly Detection...\n")

# This will identify the exact line in losses.py or hivt.py causing NaNs
try:
    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(model, datamodule)
    print("\n✅ Sanity check finished successfully (No NaNs detected)\n")
except Exception as e:
    print(f"\n❌ Sanity check failed with error:\n{e}\n")