import os
import json
import torch
import torch.nn.functional as F
from typing import Optional
from torch_geometric.data import Dataset, Batch
from utils import TemporalData

# --- MAPPING ---
MANEUVER_MAP = {
    "Straight Drive": 0, "Left Turn": 1, "Right Turn": 2, "U-Turn": 3,
    "Lane Change Left": 4, "Lane Change Right": 5, "Stationary Stop": 6, "Unknown": -1
}

LANE_TYPE_MAP = {
    "Single-lane": 0, "2-lane": 1, "3-lane": 2, "4-lane": 3, "Multi-lane": 4, "Unknown": -1
}

class NuScenesHiVTDataset(Dataset):
    def __init__(
        self,
        split_file: str,
        split: str = "train",
        tokenizer=None, 
        transform=None,
        root: str = None, 
        max_samples: Optional[int] = None,
        min_historical_steps: int = 20,
    ):
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.min_historical_steps = min_historical_steps
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        self._file_paths = splits[split]
        if max_samples is not None:
            self._file_paths = self._file_paths[:max_samples]
            
        print(f"[{split.upper()}] Loaded {len(self._file_paths)} samples.")
        super().__init__(root=None, transform=transform)

    def len(self) -> int:
        return len(self._file_paths)

    def get(self, idx: int):
        path = self._file_paths[idx]
        try:
            try:
                data = torch.load(path, weights_only=False)
            except TypeError:
                data = torch.load(path)
        except Exception:
            print(f"Corrupt file: {path}")
            return self.get((idx + 1) % len(self))

        # 1. Sanitize Data (Crop & Align)
        data = self._sanitize(data)
        
        # 2. Skip Empty Graphs
        if data.num_nodes == 0:
            return self.get((idx + 1) % len(self))

        # 3. Temporal Padding
        data = self._pad_temporal(data)

        # 4. REGENERATE ROTATE MAT (The Fix)
        # We perform this AFTER sanitization/padding to guarantee size match
        num_nodes = data.num_nodes
        if hasattr(data, 'rotate_angles') and data.rotate_angles is not None:
            # Generate from angles: [N, 2, 2]
            # Use the LAST time step (current time) for rotation
            theta = data.rotate_angles[:, -1] 
            cos, sin = theta.cos(), theta.sin()
            # Rotation matrix: [[cos, -sin], [sin, cos]]
            row1 = torch.stack([cos, -sin], dim=1)
            row2 = torch.stack([sin, cos], dim=1)
            data.rotate_mat = torch.stack([row1, row2], dim=1)
        else:
            # Fallback to Identity
            data.rotate_mat = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(num_nodes, 1, 1)

        # 5. Extract Text/Labels
        cap_dict = getattr(data, 'caption_dict', {}) if not isinstance(data, dict) else data.get('caption_dict', {})
        cat_str = getattr(data, 'maneuver_category', "Unknown")
        if isinstance(cat_str, list): cat_str = cat_str[0]
        
        full_text = f"{cap_dict.get('maneuver_type', '')} {cap_dict.get('lane_status', '')} {cap_dict.get('scene_description', '')}".strip()
        if len(full_text) < 5: full_text = "Traffic scene."

        if self.tokenizer is not None:
            enc = self.tokenizer(full_text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
            data.input_ids = enc['input_ids'].squeeze(0)
            data.attention_mask = enc['attention_mask'].squeeze(0)

        data.maneuver_id = torch.tensor([MANEUVER_MAP.get(cat_str, -1)], dtype=torch.long)
        
        l_type = cap_dict.get('lane_type', "Unknown")
        l_id = -1
        for key, val in LANE_TYPE_MAP.items():
            if key in l_type: l_id = val; break
        data.lane_type_id = torch.tensor([l_id], dtype=torch.long)

        # 6. Convert to TemporalData
        out_data = TemporalData()
        for key, value in data.to_dict().items():
            out_data[key] = value
        out_data.num_nodes = data.num_nodes
        
        return out_data

    def _pad_temporal(self, data):
        if not hasattr(data, 'x') or data.x is None: return data
        current_steps = data.x.size(1)
        if current_steps >= self.min_historical_steps: return data
        pad = self.min_historical_steps - current_steps
        
        data.x = F.pad(data.x, (0, 0, 0, pad))
        if hasattr(data, 'positions'): data.positions = F.pad(data.positions, (0, 0, 0, pad))
        if hasattr(data, 'rotate_angles'): data.rotate_angles = F.pad(data.rotate_angles, (0, pad))
        if hasattr(data, 'padding_mask'): data.padding_mask = F.pad(data.padding_mask, (0, pad), value=True)
        if hasattr(data, 'bos_mask'): data.bos_mask = F.pad(data.bos_mask, (0, pad), value=True)
        return data

    def _sanitize(self, data):
        # A. Node Count
        node_counts = []
        if hasattr(data, 'x') and torch.is_tensor(data.x): node_counts.append(data.x.size(0))
        if hasattr(data, 'positions') and torch.is_tensor(data.positions): node_counts.append(data.positions.size(0))
        valid_num_nodes = min(node_counts) if node_counts else 0
        
        for key in ['x', 'positions', 'padding_mask', 'bos_mask', 'rotate_angles', 'y']:
            if hasattr(data, key):
                tensor = getattr(data, key)
                if torch.is_tensor(tensor) and tensor.size(0) > valid_num_nodes:
                    setattr(data, key, tensor[:valid_num_nodes])
        data.num_nodes = valid_num_nodes
        
        # **FORCE DELETE rotate_mat** to ensure it's regenerated correctly later
        if hasattr(data, 'rotate_mat'): del data.rotate_mat
        
        if valid_num_nodes == 0: return data

        # B. AV Index
        if hasattr(data, 'av_index') and torch.is_tensor(data.av_index):
            if data.av_index.numel() == 1 and data.av_index.item() >= valid_num_nodes:
                data.av_index.fill_(0)
            elif data.av_index.numel() > 1:
                data.av_index = data.av_index[data.av_index < valid_num_nodes]
                if data.av_index.numel() == 0: data.av_index = torch.tensor([0], device=data.x.device)

        # C. Lane Count
        lane_counts = []
        if hasattr(data, 'lane_vectors') and torch.is_tensor(data.lane_vectors): lane_counts.append(data.lane_vectors.size(0))
        real_num_lanes = min(lane_counts) if lane_counts else 0
        
        for key in ['lane_vectors', 'is_intersections', 'turn_directions', 'traffic_controls']:
            if hasattr(data, key):
                tensor = getattr(data, key)
                if torch.is_tensor(tensor) and tensor.size(0) > real_num_lanes:
                    setattr(data, key, tensor[:real_num_lanes])

        # D. Filter Edges
        if hasattr(data, "lane_actor_index"):
            lai = data.lane_actor_index
            if real_num_lanes == 0 or not torch.is_tensor(lai) or lai.numel() == 0:
                data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
                data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)
            else:
                if lai.dim() == 1: lai = lai.reshape(2, 1)
                valid_mask = (lai[0] < real_num_lanes) & (lai[1] < valid_num_nodes) & (lai[0] >= 0) & (lai[1] >= 0)
                data.lane_actor_index = lai[:, valid_mask]
                if hasattr(data, "lane_actor_vectors") and torch.is_tensor(data.lane_actor_vectors):
                    if data.lane_actor_vectors.shape[0] == valid_mask.shape[0]:
                        data.lane_actor_vectors = data.lane_actor_vectors[valid_mask]
                    else:
                        data.lane_actor_vectors = torch.empty((data.lane_actor_index.shape[1], 2), dtype=torch.float)

        if hasattr(data, "edge_index"):
            ei = data.edge_index
            if not torch.is_tensor(ei) or ei.numel() == 0:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                if ei.dim() == 1: ei = ei.reshape(2, 1)
                valid_mask = (ei[0] < valid_num_nodes) & (ei[1] < valid_num_nodes)
                data.edge_index = ei[:, valid_mask]

        # E. Clamp Categorical
        if hasattr(data, 'is_intersections') and torch.is_tensor(data.is_intersections):
            data.is_intersections = torch.clamp(data.is_intersections, min=0, max=1)
        if hasattr(data, 'traffic_controls') and torch.is_tensor(data.traffic_controls):
            data.traffic_controls = torch.clamp(data.traffic_controls, min=0, max=1)
        if hasattr(data, 'turn_directions') and torch.is_tensor(data.turn_directions):
            data.turn_directions = torch.clamp(data.turn_directions, min=0, max=2)

        return data

    @staticmethod
    def collate_fn(batch_list):
        batch = Batch.from_data_list(batch_list)
        
        # MANUAL RE-BATCHING of Indices
        lane_actor_indices = []
        lane_actor_vectors = []
        edge_indices = []
        
        lane_offset = 0
        node_offset = 0
        
        for data in batch_list:
            if hasattr(data, 'lane_actor_index') and data.lane_actor_index.numel() > 0:
                lai = data.lane_actor_index.clone()
                lai[0] += lane_offset
                lai[1] += node_offset
                lane_actor_indices.append(lai)
                if hasattr(data, 'lane_actor_vectors'):
                    lane_actor_vectors.append(data.lane_actor_vectors)
            
            if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
                ei = data.edge_index.clone()
                ei += node_offset
                edge_indices.append(ei)

            if hasattr(data, 'lane_vectors') and data.lane_vectors is not None:
                lane_offset += data.lane_vectors.size(0)
            node_offset += data.num_nodes
            
        if len(lane_actor_indices) > 0:
            batch.lane_actor_index = torch.cat(lane_actor_indices, dim=1)
            if len(lane_actor_vectors) > 0:
                batch.lane_actor_vectors = torch.cat(lane_actor_vectors, dim=0)
        else:
            batch.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
            batch.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)

        if len(edge_indices) > 0:
            batch.edge_index = torch.cat(edge_indices, dim=1)
        else:
            batch.edge_index = torch.empty((2, 0), dtype=torch.long)
            
        return batch