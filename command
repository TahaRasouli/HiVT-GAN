python train.py   --root /mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes-hivt/   --embed_dim 128   --devices 2   --max_epochs 30   --use_gan   --lambda_adv 0.01   --lambda_r1 10.0  --critic_steps 1

 python train.py --root /mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes-hivt/ --embed_dim 128 --use_gan --ckpt_path /mount/arbeitsdaten/studenten4/rasoulta/HiVT-GAN/lightning_logs/version_52/checkpoints/epoch=0-step=28130.ckpt --train_batch_size 1 --val_batch_size 1 --devices 2 --lr 5e-5 --critic_lr 1e-4 --lambda_adv 0.01 --lambda_r1 1.0 --max_epochs 64 --monitor val_minADE --num_workers 0 --pin_memory False

Learning Rate: Start with 5e-4. HiVT is sensitive to the LR; if it's too high, the cross-attention layers might diverge.

Weight Decay: Use 1e-4 to help with the NuScenes "stationary car" bias.

Batch Size: Use the largest your GPU allows (typically 32 or 64). Since you have a lot of nodes (some scenes have 139 nodes!), monitor your VRAM closely.

Validation: Use the val split you are about to process. Your minFDE should start dropping significantly after the first 5-10 epochs.