import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score


class ManeuverClassifier(pl.LightningModule):

    def __init__(self,
                 frozen_backbone,
                 num_classes=6,
                 lr=5e-4,
                 class_weights=None,
                 num_traj_candidates=6,
                 future_steps=30,
                 id_to_class=None):

        super().__init__()

        # -----------------------------
        # Backbone
        # -----------------------------
        self.encoder = frozen_backbone

        for p in self.encoder.parameters():
            p.requires_grad=False

        self.encoder.eval()

        self.lr=lr
        self.K=num_traj_candidates
        self.future_steps=future_steps
        self.id_to_class=id_to_class

        embed_dim=self.encoder.hparams.embed_dim

        # -----------------------------
        # Trajectory encoder
        # -----------------------------
        self.traj_encoder=nn.Sequential(
            nn.Linear(future_steps*2,128),
            nn.ReLU(),
            nn.Linear(128,embed_dim)
        )

        # -----------------------------
        # Cross-attention fusion ⭐
        # -----------------------------
        self.cross_attn=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        # -----------------------------
        # Classifier
        # -----------------------------
        self.classifier=nn.Sequential(
            nn.Linear(embed_dim,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

        self.loss_fn=nn.CrossEntropyLoss(weight=class_weights)

        self.val_f1_macro=MulticlassF1Score(num_classes=num_classes,average="macro")
        self.val_f1_per_class=MulticlassF1Score(num_classes=num_classes,average=None)

    # ------------------------------------------------
    # FORWARD
    # ------------------------------------------------

    def forward(self,batch):

        # Scene encoding
        node_feat=self.encoder(batch).squeeze(0)   # [N_total,D]

        batch_index=batch.batch

        ego_indices=torch.cat([
            torch.tensor([0],device=batch_index.device),
            torch.where(batch_index[1:]!=batch_index[:-1])[0]+1
        ])

        ego_embed=node_feat[ego_indices]    # [B,D]

        B=ego_embed.size(0)

        # Generate trajectory candidates
        context_expanded=ego_embed.repeat_interleave(self.K,dim=0)

        traj_flat,_=self.encoder.decoder(context_expanded,y_gt=None)

        # decoder output = [B*K, T, 2]
        traj=traj_flat.view(B,self.K,self.future_steps,2)

        # Encode trajectories
        traj_feat=traj.reshape(B,self.K,-1)
        traj_embed=self.traj_encoder(traj_feat)   # [B,K,D]

        # Cross attention fusion ⭐
        query=ego_embed.unsqueeze(1)   # [B,1,D]
        key=traj_embed
        value=traj_embed

        fused,_=self.cross_attn(query,key,value)
        fused=fused.squeeze(1)

        logits=self.classifier(fused)

        return logits

    # -----------------------------
    # TRAIN
    # -----------------------------

    def training_step(self,batch,batch_idx):

        logits=self(batch)
        targets=batch.maneuver_id.view(-1)

        loss=self.loss_fn(logits,targets)

        self.log("train_loss",loss,prog_bar=True)
        return loss

    # -----------------------------
    # VAL
    # -----------------------------

    def validation_step(self,batch,batch_idx):

        logits=self(batch)
        targets=batch.maneuver_id.view(-1)

        preds=torch.argmax(logits,dim=1)

        self.val_f1_macro.update(preds,targets)
        self.val_f1_per_class.update(preds,targets)

    def on_validation_epoch_end(self):

        f1_macro=self.val_f1_macro.compute()
        f1_per_class=self.val_f1_per_class.compute()

        self.log("val_f1_macro",f1_macro,prog_bar=True)

        if self.global_rank==0:

            print("\n==== Per-class F1 ====")

            for i,f in enumerate(f1_per_class):
                name=self.id_to_class[i] if self.id_to_class else str(i)
                print(f"{name}: {f.item():.4f}")

        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),lr=self.lr)
