import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore 
from models.cvae_gan import CVAE_GAN
from models.captioner import TrajectoryCaptioner
from utils import SimpleTokenizer

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, vocab_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Backbone
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        self.backbone.freeze() 
        self.backbone.eval()
        
        # 2. Captioner
        self.captioner = TrajectoryCaptioner(vocab_size=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) 
        
        # 3. Metrics
        self.bleu4 = BLEUScore(n_gram=4)
        self.tokenizer = SimpleTokenizer(vocab_file="vocab.json") 

    def _get_ego_features(self, data):
        """
        Extracts only the Ego Vehicle (Node 0) features from the dense graph batch.
        """
        # 1. Get Global Context for ALL agents [Total_Nodes, 128]
        with torch.no_grad():
            all_global_embed = self.backbone(data).reshape(-1, 128)
            
        # 2. Identify Ego Indices
        # data.ptr contains the index where each graph starts in the batch.
        # The Ego vehicle is always the first node (index 0) of every graph.
        # data.ptr is [0, N1, N1+N2, ...]. We take all except the last element.
        ego_indices = data.ptr[:-1]
        
        # 3. Slice
        global_embed = all_global_embed[ego_indices] # [Batch_Size, 128]
        traj_y = data.y[ego_indices]                 # [Batch_Size, 30, 2]
        
        return global_embed, traj_y

    def forward(self, data):
        # Inference Forward Pass
        global_embed, traj_y = self._get_ego_features(data)
        return self.captioner(global_embed, traj_y)

    def training_step(self, data, batch_idx):
        # 1. Get Sliced Features (Batch Size = 64)
        global_embed, traj_y = self._get_ego_features(data)
            
        # 2. Forward with Targets
        # data.caption_ids is already [Batch_Size, Seq_Len]
        logits = self.captioner(global_embed, traj_y, captions=data.caption_ids)
        
        # 3. Loss
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), data.caption_ids.reshape(-1))
        
        self.log("train_cap_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        global_embed, traj_y = self._get_ego_features(data)
            
        # 1. Loss
        logits = self.captioner(global_embed, traj_y, captions=data.caption_ids)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), data.caption_ids.reshape(-1))
        self.log("val_cap_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        
        # 2. BLEU Score (Greedy Decode)
        pred_logits = self.captioner(global_embed, traj_y, captions=None)
        pred_ids = pred_logits.argmax(dim=-1)
        
        preds = [self.tokenizer.decode(ids) for ids in pred_ids]
        targets = [self.tokenizer.decode(ids) for ids in data.caption_ids]
        
        self.bleu4.update(preds, [[t] for t in targets])

    def on_validation_epoch_end(self):
        bleu_score = self.bleu4.compute()
        self.log("val_bleu4", bleu_score, prog_bar=True)
        self.bleu4.reset()
        
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            val_loss = metrics.get('val_cap_loss', 0.0)
            print(f"\nEpoch {self.current_epoch:03d} | Val Loss: {val_loss:.4f} | BLEU-4: {bleu_score:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.captioner.parameters(), lr=1e-3, weight_decay=1e-5)