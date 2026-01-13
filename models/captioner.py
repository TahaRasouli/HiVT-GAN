import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionalTrajectoryEncoder(nn.Module):
    """
    Encodes trajectory [B, 30, 2] -> [B, 30, 128]
    Preserves temporal dimension so Attention can look at specific steps.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=128):
        super().__init__()
        # We use a small 1D Conv or LSTM to process per-step features
        # 1D Conv is faster and good for local patterns (curves)
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
        # Permute back: [B, 30, 128]
        return x.permute(0, 2, 1)

class BahdanauAttention(nn.Module):
    """Calculates attention weights over the trajectory steps"""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: Decoder hidden state [B, 1, Hidden]
        # keys: Trajectory features [B, 30, Hidden]
        
        # Calculate Energy
        # We expand query to match keys length
        query = query.repeat(1, keys.size(1), 1) # [B, 30, Hidden]
        
        energy = torch.tanh(self.Wa(query) + self.Ua(keys)) # [B, 30, Hidden]
        scores = self.Va(energy) # [B, 30, 1]
        
        # Softmax to get weights (sum to 1)
        weights = F.softmax(scores, dim=1)
        
        # Context vector is weighted sum of keys
        context = torch.bmm(weights.permute(0, 2, 1), keys) # [B, 1, Hidden]
        return context, weights

class AttentionalCaptionDecoder(nn.Module):
    def __init__(self, context_dim, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Attention Mechanism
        self.attention = BahdanauAttention(hidden_dim)
        
        # GRU input = Embedding + Context Vector
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        
        # Project Global Context (HiVT) to init hidden state
        self.init_proj = nn.Linear(context_dim, hidden_dim)
        
        # Output Head
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, global_context, traj_feats, captions=None, teacher_forcing_ratio=0.5):
        """
        global_context: [B, 128] (Used to initialize the GRU hidden state)
        traj_feats:     [B, 30, 128] (The sequence we attend to)
        """
        batch_size = global_context.size(0)
        
        # 1. Initialize Hidden State with Global Context (Map info)
        hidden = torch.tanh(self.init_proj(global_context)).unsqueeze(0) # [1, B, Hidden]
        
        # 2. Start Token
        input_token = torch.tensor([1] * batch_size, device=global_context.device).unsqueeze(1)
        
        outputs = []
        max_len = captions.size(1) if captions is not None else 20
        
        for t in range(max_len):
            # A. Calculate Attention Context
            # Look at trajectory (traj_feats) using current hidden state
            # context: [B, 1, Hidden] (The relevant part of the trajectory for THIS word)
            context, attn_weights = self.attention(hidden.permute(1, 0, 2), traj_feats)
            
            # B. Get Word Embedding
            embedded = self.dropout(self.embedding(input_token)) # [B, 1, Emb]
            
            # C. Concatenate Embedding + Context
            gru_input = torch.cat([embedded, context], dim=2)
            
            # D. GRU Step
            output, hidden = self.gru(gru_input, hidden)
            
            # E. Predict
            prediction = self.fc_out(output.squeeze(1))
            outputs.append(prediction.unsqueeze(1))
            
            # F. Next Token
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = captions[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1).unsqueeze(1)
                
        return torch.cat(outputs, dim=1)

class TrajectoryCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Use the new Attentional components
        self.traj_encoder = AttentionalTrajectoryEncoder(hidden_dim=64, output_dim=256)
        self.decoder = AttentionalCaptionDecoder(
            context_dim=128, 
            embed_dim=128, 
            hidden_dim=256, 
            vocab_size=vocab_size
        )
        
    def forward(self, global_context, traj, captions=None):
        traj_feats = self.traj_encoder(traj) # Returns sequence [B, 30, 256]
        return self.decoder(global_context, traj_feats, captions)