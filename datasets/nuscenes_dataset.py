import os
import json
import torch
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
    ):
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        
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

        # 1. Sanitize Data
        data = self._sanitize(data)
        
        # SKIP EMPTY GRAPHS
        if data.num_nodes == 0:
            return self.get((idx + 1) % len(self))

        # 2. Inject Rotate Mat if missing
        if not hasattr(data, 'rotate_mat') or data.rotate_mat is None:
            num_nodes = data.num_nodes 
            identity_rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(num_nodes, 1, 1)
            data.rotate_mat = identity_rot

        # 3. Extract Text/Labels
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
        
        # 4. Use standard TemporalData (We handle batching in collate_fn now)
        out_data = TemporalData()
        for key, value in data.to_dict().items():
            out_data[key] = value
        out_data.num_nodes = data.num_nodes
        return out_data

    def _sanitize(self, data):
        # A. Node Count Alignment
        node_counts = []
        if hasattr(data, 'x') and torch.is_tensor(data.x): node_counts.append(data.x.size(0))
        if hasattr(data, 'positions') and torch.is_tensor(data.positions): node_counts.append(data.positions.size(0))
        
        valid_num_nodes = min(node_counts) if node_counts else 0
        
        # Crop Node Tensors
        for key in ['x', 'positions', 'padding_mask', 'bos_mask', 'rotate_angles', 'y']:
            if hasattr(data, key):
                tensor = getattr(data, key)
                if torch.is_tensor(tensor) and tensor.size(0) > valid_num_nodes:
                    setattr(data, key, tensor[:valid_num_nodes])
        data.num_nodes = valid_num_nodes
        if valid_num_nodes == 0: return data

        # B. AV Index
        if hasattr(data, 'av_index') and torch.is_tensor(data.av_index):
            if data.av_index.numel() == 1 and data.av_index.item() >= valid_num_nodes:
                data.av_index.fill_(0)
            elif data.av_index.numel() > 1:
                data.av_index = data.av_index[data.av_index < valid_num_nodes]
                if data.av_index.numel() == 0: data.av_index = torch.tensor([0], device=data.x.device)

        # C. Lane Count Alignment
        lane_counts = []
        if hasattr(data, 'lane_vectors') and torch.is_tensor(data.lane_vectors): lane_counts.append(data.lane_vectors.size(0))
        if hasattr(data, 'is_intersections') and torch.is_tensor(data.is_intersections): lane_counts.append(data.is_intersections.size(0))
        
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
        # 1. Standard PyG Batching (Handles x, positions, edge_index, etc.)
        batch = Batch.from_data_list(batch_list)
        
        # 2. MANUAL FIX for Lane-Actor Index (The Source of Crashes)
        # We manually calculate offsets to guarantee correctness
        lane_actor_indices = []
        lane_actor_vectors = []
        lane_offset = 0
        node_offset = 0
        
        for data in batch_list:
            # Shift Indices
            if hasattr(data, 'lane_actor_index') and data.lane_actor_index.numel() > 0:
                lai = data.lane_actor_index.clone()
                lai[0] += lane_offset # Row 0: Lanes
                lai[1] += node_offset # Row 1: Actors
                lane_actor_indices.append(lai)
                
                if hasattr(data, 'lane_actor_vectors'):
                    lane_actor_vectors.append(data.lane_actor_vectors)
            
            # Update Offsets
            if hasattr(data, 'lane_vectors') and data.lane_vectors is not None:
                lane_offset += data.lane_vectors.size(0)
            node_offset += data.num_nodes
            
        # Overwrite the batched index with our manually calculated one
        if len(lane_actor_indices) > 0:
            batch.lane_actor_index = torch.cat(lane_actor_indices, dim=1)
            if len(lane_actor_vectors) > 0:
                batch.lane_actor_vectors = torch.cat(lane_actor_vectors, dim=0)
        else:
            batch.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
            batch.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)
            
        return batch