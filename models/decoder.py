from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):
    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        # --- NEW: Mode-specific Latent Code ---
        # This allows the GAN to differentiate the 6 branches
        self.mode_emb = nn.Parameter(torch.randn(num_modes, local_channels) * 0.01)

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
            
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
            
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
                
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Inject mode-specific latent info into the expanded local_embed
        # [F, N, D] = [F, 1, D] + [1, N, D]
        # This keeps the backbone weights valid but differentiates the modes for the GAN
        expanded_local = local_embed.unsqueeze(0).expand(self.num_modes, -1, -1)
        mode_specific_local = expanded_local + self.mode_emb.unsqueeze(1)

        # 2. Probability (pi) uses the modified embedding
        pi = self.pi(torch.cat((mode_specific_local, global_embed), dim=-1)).squeeze(-1).t()
        
        # 3. Aggregated embedding
        out = self.aggr_embed(torch.cat((global_embed, mode_specific_local), dim=-1))
        
        # 4. Trajectory Generation
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)
        
        if self.uncertain:
            # We use the clamp here for stability as discussed before
            scale = F.elu(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = torch.clamp(scale + self.min_scale, min=0.05) 
            return torch.cat((loc, scale), dim=-1), pi
        else:
            return loc, pi


class CVAEDecoder(nn.Module):
    def __init__(self, embed_dim=128, latent_dim=16, future_steps=30, num_modes=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        
        # 1. Posterior & 2. Prior (Keep existing code...)
        self.posterior_net = nn.Sequential(
            nn.Linear(embed_dim + future_steps * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim * 2)
        )
        
        self.prior_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim * 2)
        )
        
        # 3. Generator Feature Extractor (Shared)
        # We split the original generator so we can access the hidden state
        self.generator_features = nn.Sequential(
            nn.Linear(embed_dim + latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
            # We stop here to fork
        )

        # 4. Trajectory Head (The original output layer)
        self.trajectory_head = nn.Linear(embed_dim, future_steps * 2)

        # 5. [NEW] Caption Head (The Superfast Classifier)
        # Outputs 7 scores (one for each maneuver class)
        self.caption_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 7)
        )

    def forward(self, context, y_gt=None):
        """
        context: [Batch, Embed_Dim]
        y_gt: [Batch, Future_Steps, 2] (Optional, only for training CVAE)
        """
        # A. Prior / Posterior Logic (Keep existing...)
        prior_out = self.prior_net(context)
        prior_mu, prior_logvar = prior_out.chunk(2, dim=-1)
        
        z = None
        kld_loss = torch.tensor(0.0, device=context.device)
        
        if self.training and y_gt is not None:
            y_flat = y_gt.reshape(y_gt.size(0), -1)
            post_input = torch.cat([context, y_flat], dim=-1)
            post_out = self.posterior_net(post_input)
            post_mu, post_logvar = post_out.chunk(2, dim=-1)
            std = torch.exp(0.5 * post_logvar)
            eps = torch.randn_like(std)
            z = post_mu + eps * std
            kld_loss = -0.5 * torch.sum(1 + post_logvar - prior_logvar - 
                                      (post_mu - prior_mu).pow(2) / torch.exp(prior_logvar) - 
                                      torch.exp(post_logvar) / torch.exp(prior_logvar), dim=1).mean()
        else:
            std = torch.exp(0.5 * prior_logvar)
            eps = torch.randn_like(std)
            z = prior_mu + eps * std
            
        # B. Generator Forward Pass
        gen_input = torch.cat([context, z], dim=-1)
        
        # 1. Get Shared Features
        features = self.generator_features(gen_input)
        
        # 2. Head A: Trajectory
        y_hat = self.trajectory_head(features)
        y_hat = y_hat.reshape(-1, self.future_steps, 2)
        
        # 3. [NEW] Head B: Caption
        caption_logits = self.caption_head(features) # [Batch, 7]
        
        return y_hat, kld_loss, caption_logits