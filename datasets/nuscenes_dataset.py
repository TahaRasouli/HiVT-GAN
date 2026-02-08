import os
from typing import Optional, List, Dict
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData
from tqdm import tqdm

class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset with class-balanced subsampling.
    
    Excludes U-Turns (ID 3) and caps Straight (ID 0) and Stationary (ID 6).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
    ):
        self.split = split
        self.root = root
        self.transform = transform
        self._directory = f"{split}_processed"

        # 1. Path Detection
        standard_path = os.path.join(self.root, self._directory)
        if os.path.isdir(standard_path):
            self._processed_dir = standard_path
            self.is_flat = False
        else:
            self._processed_dir = self.root
            self.is_flat = True

        # 2. File Listing
        all_files = sorted(
            f for f in os.listdir(self._processed_dir) if f.endswith(".pt")
        )

        # 3. Class-Based Filtering & Capping (Training Split Only)
        if split == "train":
            print(f"[Dataset] Filtering training data (Caps: Straight=500, Stat=300)...")
            
            # Target limits: 0: Straight, 6: Stationary. ID 3 (U-Turn) is excluded.
            limits = {0: 500, 6: 300}
            counters = {0: 0, 6: 0}
            filtered_files = []

            for f in tqdm(all_files, desc="Filtering Samples"):
                # Load metadata without full graph if possible (torch.load is necessary)
                data = torch.load(os.path.join(self._processed_dir, f))
                
                # Use maneuver_type or maneuver_id if already saved
                m_id = getattr(data, 'maneuver_id', None)
                if m_id is None:
                    # Fallback to sanitization logic to find ID
                    m_id = self._get_maneuver_id(data)
                else:
                    m_id = int(m_id.item())

                # Exclude U-Turns
                if m_id == 3:
                    continue
                
                # Apply Capping
                if m_id in limits:
                    if counters[m_id] < limits[m_id]:
                        filtered_files.append(f)
                        counters[m_id] += 1
                else:
                    # Keep all Turns and Lane Changes
                    filtered_files.append(f)
            
            self._processed_file_names = filtered_files
            print(f"[Dataset] Filter complete. Final training set: {len(self._processed_file_names)} samples.")
        else:
            # For validation, we use standard split logic
            if self.is_flat:
                num_total = len(all_files)
                num_val = int(num_total * val_ratio)
                num_train = num_total - num_val
                self._processed_file_names = all_files[num_train:]
            else:
                self._processed_file_names = all_files

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    # --------------------------------------------------------------
    # INTERNAL UTILITIES
    # --------------------------------------------------------------
    def _get_maneuver_id(self, data: TemporalData) -> int:
        """Helper to derive ID from attributes saved during labeling."""
        str_to_int = {
            "follow": 0, "Straight": 0,
            "turn_left": 1, "Left Turn": 1,
            "turn_right": 2, "Right Turn": 2,
            "u_turn": 3, "U-Turn": 3,
            "lane_change_left": 4,
            "lane_change_right": 5,
            "stationary": 6, "Stationary": 6
        }
        
        # Try different possible attribute names from previous scripts
        label = getattr(data, 'maneuver_type', None) or getattr(data, 'maneuver_category', "follow")
        return str_to_int.get(label, 0)

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    # --------------------------------------------------------------
    # SANITIZATION
    # --------------------------------------------------------------
    def _sanitize(self, data: TemporalData) -> TemporalData:
        # 1. Range-Safe Graph Indices
        if hasattr(data, "lane_actor_index") and hasattr(data, "lane_vectors"):
            lai = data.lane_actor_index
            num_lanes = data.lane_vectors.size(0)
            num_actors = data.num_nodes
            valid = (lai[0] >= 0) & (lai[0] < num_lanes) & (lai[1] >= 0) & (lai[1] < num_actors)
            data.lane_actor_index = lai[:, valid]

        if hasattr(data, "edge_index"):
            ei = data.edge_index
            num_nodes = data.num_nodes
            valid = (ei[0] >= 0) & (ei[0] < num_nodes) & (ei[1] >= 0) & (ei[1] < num_nodes)
            data.edge_index = ei[:, valid]

        # 2. Vector Shape Fixes
        for key in ["lane_actor_vectors", "lane_vectors"]:
            if hasattr(data, key):
                vec = getattr(data, key)
                if not torch.is_tensor(vec) or vec.numel() == 0 or vec.dim() != 2:
                    setattr(data, key, torch.empty((0, 2), dtype=torch.float))

        # 3. Feature Clamping
        for attr in ["turn_directions", "traffic_controls", "is_intersections"]:
            if hasattr(data, attr):
                val = getattr(data, attr)
                setattr(data, attr, torch.clamp(val, 0, 2).long())

        # 4. Maneuver ID Assignment
        m_id = self._get_maneuver_id(data)
        data.maneuver_id = torch.tensor([m_id], dtype=torch.long)

        # 5. Ego Index
        data.ego_index = torch.tensor([0], dtype=torch.long)

        if data.num_nodes == 0:
            raise ValueError("Invalid sample: graph has zero nodes")

        return data

    # --------------------------------------------------------------
    # DATASET INTERFACE
    # --------------------------------------------------------------
    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> TemporalData:
        path = os.path.join(self.processed_dir, self._processed_file_names[idx])
        try:
            data = torch.load(path)
            return self._sanitize(data)
        except Exception:
            return self.get((idx + 1) % len(self._processed_file_names))

    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)