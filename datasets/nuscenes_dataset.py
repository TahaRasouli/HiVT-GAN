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


class HiVTTemporalData(TemporalData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            # Bipartite graph: 
            # Row 0 (Lanes) -> Inc by num_lanes
            # Row 1 (Actors) -> Inc by num_nodes
            # Returns shape [2, 1] for broadcasting
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        elif key == 'edge_index':
            # Agent-Agent graph:
            # Both rows are Actors -> Inc by num_nodes
            return self.num_nodes
        elif 'edge_index' in key:
            # Catch-all for temporal edges like 'edge_index_0', 'edge_index_1' etc.
            return self.num_nodes
        else:
            # Default for everything else
            return super().__inc__(key, value, *args, **kwargs)

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
            # Load with weights_only=False to support custom objects
            try:
                data = torch.load(path, weights_only=False)
            except TypeError:
                data = torch.load(path)
        except Exception:
            print(f"Corrupt file: {path}")
            return self.get((idx + 1) % len(self))

        # 1. Sanitize Data (Remove bad indices)
        data = self._sanitize(data)
        
        # 2. Inject Rotate Mat if missing
        if not hasattr(data, 'rotate_mat') or data.rotate_mat is None:
            num_nodes = data.num_nodes 
            identity_rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(num_nodes, 1, 1)
            data.rotate_mat = identity_rot

        # 3. Extract Text/Labels
        cap_dict = getattr(data, 'caption_dict', {}) if not isinstance(data, dict) else data.get('caption_dict', {})
        
        man_text = cap_dict.get('maneuver_type', "")
        lane_text = cap_dict.get('lane_status', "")
        scene_desc = cap_dict.get('scene_description', "")
        
        cat_str = getattr(data, 'maneuver_category', "Unknown")
        if isinstance(cat_str, list): cat_str = cat_str[0]
        
        full_text = f"{man_text} {lane_text} {scene_desc}".strip()
        if len(full_text) < 5: full_text = "Traffic scene."

        if self.tokenizer is not None:
            enc = self.tokenizer(
                full_text, return_tensors='pt', padding='max_length', truncation=True, max_length=64 
            )
            data.input_ids = enc['input_ids'].squeeze(0)
            data.attention_mask = enc['attention_mask'].squeeze(0)

        m_id = MANEUVER_MAP.get(cat_str, -1)
        data.maneuver_id = torch.tensor([m_id], dtype=torch.long)
        
        l_type = cap_dict.get('lane_type', "Unknown")
        l_id = -1
        for key, val in LANE_TYPE_MAP.items():
            if key in l_type: l_id = val; break
        data.lane_type_id = torch.tensor([l_id], dtype=torch.long)
        
        # --- 4. CONVERT TO HiVTTemporalData ---
        # We must cast the data object to our custom class to enable the 
        # correct __inc__ logic during batching.
        safe_data = HiVTTemporalData()
        for key, value in data.to_dict().items():
            safe_data[key] = value
            
        # Ensure num_nodes is carried over explicitly
        safe_data.num_nodes = data.num_nodes
        
        return safe_data

    def _sanitize(self, data):
        """
        Sanitizes input data to ensure index validity.
        """
        # 1. ESTABLISH TRUTH
        if hasattr(data, 'x') and torch.is_tensor(data.x):
            real_num_nodes = data.x.size(0)
        else:
            real_num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else 0
        data.num_nodes = real_num_nodes

        if hasattr(data, 'lane_vectors') and torch.is_tensor(data.lane_vectors):
            real_num_lanes = data.lane_vectors.size(0)
        else:
            real_num_lanes = 0
            data.lane_vectors = torch.empty((0, 2), dtype=torch.float)

        # 2. SANITIZE LANE-ACTOR INDICES
        if hasattr(data, "lane_actor_index"):
            lai = data.lane_actor_index
            
            if real_num_lanes == 0 or not torch.is_tensor(lai) or lai.numel() == 0:
                data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
                data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)
            else:
                if lai.dim() == 1: lai = lai.reshape(2, 1)
                
                # Row 0 < num_lanes, Row 1 < num_nodes
                mask_lanes = (lai[0] < real_num_lanes) & (lai[0] >= 0)
                mask_actors = (lai[1] < real_num_nodes) & (lai[1] >= 0)
                valid_mask = mask_lanes & mask_actors
                
                data.lane_actor_index = lai[:, valid_mask]
                
                if hasattr(data, "lane_actor_vectors") and torch.is_tensor(data.lane_actor_vectors):
                    if data.lane_actor_vectors.shape[0] == valid_mask.shape[0]:
                        data.lane_actor_vectors = data.lane_actor_vectors[valid_mask]
                    else:
                        data.lane_actor_vectors = torch.empty((data.lane_actor_index.shape[1], 2), dtype=torch.float)

        # 3. SANITIZE EDGE_INDEX
        if hasattr(data, "edge_index"):
            ei = data.edge_index
            if not torch.is_tensor(ei) or ei.numel() == 0:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                if ei.dim() == 1: ei = ei.reshape(2, 1)
                mask_src = (ei[0] < real_num_nodes) & (ei[0] >= 0)
                mask_dst = (ei[1] < real_num_nodes) & (ei[1] >= 0)
                data.edge_index = ei[:, mask_src & mask_dst]

        return data

    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)