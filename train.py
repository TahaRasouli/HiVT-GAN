import torch.utils._pytree
import sys
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

# Patch for newer PyTorch versions
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node
try:
    import lightning_utilities
    if not hasattr(lightning_utilities, 'module_available'):
        from lightning_utilities.core.imports import module_available
        lightning_utilities.module_available = module_available
except ImportError:
    pass

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT
from models.cvae_gan import CVAE_GAN

# --- IMPORT YOUR NEW MODULE ---
from models.caption import CaptionFinetuner 
# ------------------------------

torch.set_float32_matmul_precision('medium')
mp.set_start_method('spawn', force=True)

def main():
    pl.seed_everything(2022)
    parser = ArgumentParser()

    # --- Data Arguments ---
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    
    # --- Training Arguments ---
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val_minFDE")
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=None)
    
    # --- Mode Flags ---
    parser.add_argument("--train_cvae_gan", action="store_true")
    
    # This flag now switches the ENTIRE logic to use your CaptionFinetuner
    parser.add_argument("--finetune_caption", action="store_true", help="Use models/caption.py logic")

    # Add model specific args (HiVT base)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    # 1. SETUP DATAMODULE
    datamodule = NuScenesHiVTDataModule(
        split_file="balanced_splits.json", 
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        tokenizer=None 
    )

    # 2. INITIALIZE MODEL
    actual_fit_path = args.ckpt_path

    # --- CASE A: CAPTION FINE-TUNING (Using models/caption.py) ---
    if args.finetune_caption:
        print(f"--- Mode: Superfast Caption Finetuning ---")
        if not args.ckpt_path:
            raise ValueError("You must provide --ckpt_path to finetune the caption head!")
            
        # Initialize your wrapper class
        # It handles loading, freezing, and strict=False inside its __init__
        model = CaptionFinetuner(pretrained_ckpt=args.ckpt_path)
        
        # We start a fresh training session for the head
        actual_fit_path = None 
        args.monitor = "val_acc"
        print("(!) Monitor set to 'val_acc'")

    # --- CASE B: CVAE_GAN ---
    elif args.train_cvae_gan:
        print("--- Mode: CVAE_GAN ---")
        args.num_modes = 1 
        model = CVAE_GAN(**vars(args))
        
        # Handle resume logic normally
        if args.ckpt_path:
             print(f"Loading weights from {args.ckpt_path}")
             # We use strict=False here too, just in case you are loading an old checkpoint
             # into the new code that has the caption_head defined (even if unused).
             ckpt = torch.load(args.ckpt_path, map_location="cpu")
             model.load_state_dict(ckpt['state_dict'], strict=False)
             # If we loaded manually, we might reset actual_fit_path to None 
             # unless you want to resume the optimizer state too.
             actual_fit_path = None 

    # --- CASE C: Standard HiVT ---
    else:
        print("--- Mode: Standard HiVT ---")
        model = HiVT(**vars(args))
        if args.ckpt_path:
             ckpt = torch.load(args.ckpt_path, map_location="cpu")
             model.load_state_dict(ckpt['state_dict'], strict=False)
             actual_fit_path = None

    # 3. TRAINER
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="max" if "acc" in args.monitor.lower() else "min",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        precision="32-true", 
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=50,
        num_sanity_val_steps=0
    )

    trainer.fit(model, datamodule, ckpt_path=actual_fit_path)

if __name__ == "__main__":
    main()