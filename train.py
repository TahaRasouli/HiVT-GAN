from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch.multiprocessing as mp
import torch

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.cvae import CVAE
from models.hivt import HiVT

# speed boost on Nvidia-A6000
torch.set_float32_matmul_precision('medium')
mp.set_start_method('spawn', force=True)

def main():
    pl.seed_everything(2022)
    parser = ArgumentParser()

    # Data arguments
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)

    # Training arguments
    parser.add_argument("--train_cvae", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val_minFDE", choices=["val_minADE", "val_minFDE", "val_minMR"])
    parser.add_argument("--save_top_k", type=int, default=5)

    # HiVT model specific args
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    # 1. Lower the Learning Rate for fine-tuning if a checkpoint is provided
    if args.ckpt_path:
        print(f"Fine-tuning detected. Lowering Learning Rate to 1e-4")
        args.lr = 1e-4 

    # 2. Model Initialization
    if args.train_cvae:
        print("--- initializing HiVT_CVAE ---")
        args.num_modes = 1 # Force 1 mode for CVAE
        model = CVAE(**vars(args))

    else: 
        print(f"--- initializing Standard HiVT ---") # Good practice to verify
        model = HiVT(**vars(args))

    # --- WARM START LOGIC (Improved) ---
    actual_fit_path = args.ckpt_path
    if args.ckpt_path:
        print(f"--- Loading Weights from: {args.ckpt_path} ---")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        
        # Check for GAN critics
        has_critics = any("D_short" in k for k in ckpt['state_dict'].keys())
        
        if not has_critics:
            # Official HiVT checkpoints are Supervised-only
            print("Detected Supervised checkpoint. Loading state_dict (strict=False).")
            model.load_state_dict(ckpt['state_dict'], strict=False)
            # Reset actual_fit_path so Lightning doesn't try to resume optimizer states
            actual_fit_path = None 
        else:
            print("Detected GAN checkpoint. Full resume enabled.")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
    )

    # Trainer
    strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        precision="16-mixed",  
        # gradient_clip_val=0.5,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback], # Critical to include this
        log_every_n_steps=50,
    )

    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    trainer.fit(model, datamodule, ckpt_path=actual_fit_path)

if __name__ == "__main__":
    main()