import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionalTrajectoryEncoder(nn.Module):
    """
    Encodes trajectory [B, 30, 2] -> [B, 30, 256]
    Preserves temporal dimension so Attention can look at specific steps.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=256):
        super().__init__()
        # 1D Conv processes temporal sequence locally
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, traj):
        # traj: [B, 30, 2] -> Permute for Conv1d: [B, 2, 30]
        x = traj.permute(0, 2, 1)
        x = self.net(x)
        # Permute back: [B, 30, output_dim]
        return x.permute(0, 2, 1)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [B, 1, Hidden]
        # keys:  [B, 30, Hidden]
        
        # Expand query to match keys length for addition
        query = query.repeat(1, keys.size(1), 1) 
        
        # Score = V * tanh(W*query + U*keys)
        energy = torch.tanh(self.Wa(query) + self.Ua(keys)) 
        scores = self.Va(energy) # [B, 30, 1]
        
        weights = F.softmax(scores, dim=1) # [B, 30, 1]
        
        # Context = Weighted Sum of Keys
        context = torch.bmm(weights.permute(0, 2, 1), keys) # [B, 1, Hidden]
        return context, weights

class AttentionalCaptionDecoder(nn.Module):
    def __init__(self, context_dim, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        
        # GRU Input = Embedding (Word) + Context (Trajectory Focus)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        
        # Project Global Context (Map info) to initialize hidden state
        self.init_proj = nn.Linear(context_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, global_context, traj_feats, captions=None, teacher_forcing_ratio=0.5, return_attn=False):
        batch_size = global_context.size(0)
        
        # 1. Init Hidden State with Map Context
        hidden = torch.tanh(self.init_proj(global_context)).unsqueeze(0) # [1, B, Hidden]
        
        # 2. Start Token
        input_token = torch.tensor([1] * batch_size, device=global_context.device).unsqueeze(1)
        
        outputs = []
        attn_list = []
        max_len = captions.size(1) if captions is not None else 20
        
        for t in range(max_len):
            # A. Attention
            # Use current hidden state to decide which part of trajectory to look at
            context, attn_weights = self.attention(hidden.permute(1, 0, 2), traj_feats)
            
            if return_attn:
                attn_list.append(attn_weights.squeeze(2)) # [B, 30]
            
            # B. GRU Step
            embedded = self.dropout(self.embedding(input_token)) 
            gru_input = torch.cat([embedded, context], dim=2)
            
            output, hidden = self.gru(gru_input, hidden)
            
            # C. Prediction
            prediction = self.fc_out(output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            
            # D. Next Token
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = captions[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1).unsqueeze(1)
                
        logits = torch.cat(outputs, dim=1)
        
        if return_attn:
            return logits, torch.stack(attn_list, dim=1) # [B, Seq_Len, 30]
        return logits

class TrajectoryCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Encoder outputs 256 dim features per step
        self.traj_encoder = AttentionalTrajectoryEncoder(output_dim=256)
        
        # Decoder expects hidden_dim=256 to match encoder output
        self.decoder = AttentionalCaptionDecoder(
            context_dim=128, 
            embed_dim=128, 
            hidden_dim=256, 
            vocab_size=vocab_size
        )
        
    def forward(self, global_context, traj, captions=None, return_attn=False):
        # traj: [B, 30, 2]
        traj_feats = self.traj_encoder(traj) # [B, 30, 256]
        return self.decoder(global_context, traj_feats, captions, return_attn=return_attn)