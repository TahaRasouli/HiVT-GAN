import os
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData
from tqdm import tqdm
from typing import Optional, List, Dict, Callable

class NuScenesHiVTDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, max_samples: Optional[int] = None):
        self.split = split
        self.root = root
        self.transform = transform
        self._processed_dir = os.path.join(self.root, f"{split}_processed")

        # 1. File Listing
        all_files = sorted(f for f in os.listdir(self._processed_dir) if f.endswith(".pt"))

        # 2. Strict Filtering (Applied to BOTH splits)
        print(f"[Dataset] Filtering {split} set (Removing U-Turns/Off-Map)...")
        
        # Training caps: 0=Straight, 6=Stationary
        caps = {0: 500, 6: 300} if split == "train" else {}
        counters = {0: 0, 6: 0}
        filtered_files = []

        for f in tqdm(all_files, desc=f"Scanning {split}"):
            data = torch.load(os.path.join(self._processed_dir, f))
            m_id = self._get_maneuver_id(data)
            
            # Exclude U-Turns (3) and Off-Map (-1)
            if m_id == 3 or m_id == -1:
                continue
            
            # Apply caps for training
            if m_id in caps:
                if counters[m_id] < caps[m_id]:
                    filtered_files.append(f)
                    counters[m_id] += 1
            else:
                filtered_files.append(f)
        
        self._processed_file_names = filtered_files
        super().__init__(root, transform=transform)

    def _get_maneuver_id(self, data) -> int:
        mapping = {"follow": 0, "turn_left": 1, "turn_right": 2, "u_turn": 3, 
                   "lane_change_left": 4, "lane_change_right": 5, "stationary": 6, "off_map": -1}
        label = getattr(data, 'maneuver_type', "follow")
        return mapping.get(label, 0)

    @property
    def processed_dir(self) -> str: return self._processed_dir
    @property
    def processed_file_names(self) -> List[str]: return self._processed_file_names

    def _sanitize(self, data: TemporalData) -> TemporalData:
        data.ego_index = torch.tensor([0], dtype=torch.long)
        data.maneuver_id = torch.tensor([self._get_maneuver_id(data)], dtype=torch.long)
        # Add basic shape fixes here if needed
        return data

    def len(self) -> int: return len(self._processed_file_names)
    def get(self, idx: int) -> TemporalData:
        return self._sanitize(torch.load(os.path.join(self.processed_dir, self._processed_file_names[idx])))