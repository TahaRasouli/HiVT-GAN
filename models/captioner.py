import torch
import torch.nn as nn
import torch.nn.functional as F

class LaneAwareTrajectoryEncoder(nn.Module):
    """
    Simplified Architecture:
    1. Encodes Raw Trajectory Geometry [B, 30, 2].
    2. Fuses with Map Context using Cross-Attention.
    """
    def __init__(self, input_dim=2, hidden_dim=64, context_dim=128, output_dim=256):
        super().__init__()
        
        # 1. Geometry Encoder (Implicitly learns heading/velocity via Conv1d)
        self.geo_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Fusion Layer (Cross Attention)
        # Query: Trajectory
        # Key/Value: Lane/Map Data
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True,
            kdim=context_dim, 
            vdim=context_dim
        )
        
        # 3. Output Projection
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, traj, global_context):
        # traj: [B, 30, 2] (Raw coordinates)
        # global_context: [B, 128] (Encoded Map/Lane data)
        
        # A. Encode Geometry
        x_geo = traj.permute(0, 2, 1) # [B, 2, 30]
        x_geo = self.geo_net(x_geo)   # [B, Hidden, 30]
        x_geo = x_geo.permute(0, 2, 1) # [B, 30, Hidden]
        
        # B. Prepare Map Context
        # Treat the Global Context vector as a single distinct feature
        context = global_context.unsqueeze(1) # [B, 1, 128]
        
        # C. FUSE: Trajectory attends to Map
        # "For every step of my path, which part of the map is relevant?"
        attn_out, _ = self.multihead_attn(
            query=x_geo, 
            key=context, 
            value=context
        )
        
        # D. Residual Connection (Geometry + Context)
        x_fused = x_geo + attn_out 
        x_out = self.out_proj(x_fused) # [B, 30, 256]
        
        return self.norm(x_out)

class BahdanauAttention(nn.Module):
    # Standard Attention for the Language Decoder
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        query = query.repeat(1, keys.size(1), 1) 
        energy = torch.tanh(self.Wa(query) + self.Ua(keys)) 
        scores = self.Va(energy)
        weights = F.softmax(scores, dim=1) 
        context = torch.bmm(weights.permute(0, 2, 1), keys)
        return context, weights

class AttentionalCaptionDecoder(nn.Module):
    def __init__(self, context_dim, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.init_proj = nn.Linear(context_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, global_context, traj_feats, captions=None, teacher_forcing_ratio=0.5, return_attn=False):
        batch_size = global_context.size(0)
        
        # Initialize decoder state with the Map Context
        hidden = torch.tanh(self.init_proj(global_context)).unsqueeze(0)
        input_token = torch.tensor([1] * batch_size, device=global_context.device).unsqueeze(1)
        
        outputs = []
        attn_list = []
        max_len = captions.size(1) if captions is not None else 20
        
        for t in range(max_len):
            # Attention looks at the FUSED trajectory features
            context, attn_weights = self.attention(hidden.permute(1, 0, 2), traj_feats)
            
            if return_attn:
                attn_list.append(attn_weights.squeeze(2))
            
            embedded = self.dropout(self.embedding(input_token)) 
            gru_input = torch.cat([embedded, context], dim=2)
            output, hidden = self.gru(gru_input, hidden)
            prediction = self.fc_out(output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = captions[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1).unsqueeze(1)
                
        logits = torch.cat(outputs, dim=1)
        if return_attn:
            return logits, torch.stack(attn_list, dim=1)
        return logits

class TrajectoryCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 1. Lane-Aware Encoder (Fusion)
        self.traj_encoder = LaneAwareTrajectoryEncoder(
            input_dim=2, # Raw (x,y)
            hidden_dim=64, 
            context_dim=128, 
            output_dim=256
        )
        
        # 2. Decoder
        self.decoder = AttentionalCaptionDecoder(
            context_dim=128, 
            embed_dim=128, 
            hidden_dim=256, 
            vocab_size=vocab_size
        )
        
    def forward(self, global_context, traj, captions=None, return_attn=False):
        # We pass the global_context to BOTH the encoder (for fusion) and decoder (for init)
        traj_feats = self.traj_encoder(traj, global_context) # [B, 30, 256]
        return self.decoder(global_context, traj_feats, captions, return_attn=return_attn)