from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch.multiprocessing as mp
import torch
import warnings

# Filter batch size warnings
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size`.*")

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT
from models.divt import DiVT

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
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val_minFDE", choices=["val_minADE", "val_minFDE", "val_minMR"])
    parser.add_argument("--save_top_k", type=int, default=5)

    # NEW: Diffusion Trigger
    parser.add_argument("--train_diffusion", action="store_true", help="Enable Diffusion Training Mode")
    parser.add_argument("--diff_steps", type=int, default=100, help="Number of diffusion steps")

    # Load Base HiVT Arguments (Shared by both models)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    # 1. Lower the Learning Rate for fine-tuning if a checkpoint is provided
    if args.ckpt_path:
        print(f"Fine-tuning/Resume detected. Lowering Learning Rate to 1e-4")
        args.lr = 1e-4 

    # 2. Model Initialization (Branch Logic)
    if args.train_diffusion:
        print("--- initializing DiVT Model ---")
        # FORCE num_modes=1 for Diffusion
        # This tells GlobalInteractor to output a single context vector [1, N, 128]
        args.num_modes = 1 
        model = DiVT(**vars(args))
    else:
        print(f"--- initializing Standard HiVT (GAN={args.use_gan}) ---")
        model = HiVT(**vars(args))

    # --- WARM START LOGIC (Modified for Transfer Learning) ---
    actual_fit_path = args.ckpt_path
    
    if args.ckpt_path:
        print(f"--- Loading Weights from: {args.ckpt_path} ---")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt['state_dict']
        
        # Scenario A: Diffusion Training (Transfer Learning from Supervised)
        if args.train_diffusion:
            print("Transferring Encoder weights to Diffusion Model...")
            # We assume the checkpoint is Standard HiVT. The encoders match, but decoder differs.
            # strict=False allows us to load Local/Global encoders and ignore the missing MLPDecoder
            model.load_state_dict(state_dict, strict=False)
            
            # We must reset fit_path so Lightning creates a FRESH optimizer 
            # (Diffusion needs its own AdamW, not the old one)
            actual_fit_path = None
            
        # Scenario B: Standard Resume (Supervised or GAN)
        else:
            has_critics = any("D_short" in k for k in state_dict.keys())
            
            if not has_critics:
                # Supervised -> GAN or Supervised -> Supervised
                print("Detected Supervised checkpoint. Loading state_dict (strict=False).")
                model.load_state_dict(state_dict, strict=False)
                actual_fit_path = None 
            else:
                # GAN -> GAN Resume
                print("Detected GAN checkpoint. Full resume enabled.")
                # Keep actual_fit_path so Lightning resumes optimizer state

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
    )

    # Trainer
    # Note: find_unused_parameters=True is required for GANs, but harmless for Diffusion
    strategy = DDPStrategy(find_unused_parameters=True)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        precision="16-mixed",  
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback], 
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