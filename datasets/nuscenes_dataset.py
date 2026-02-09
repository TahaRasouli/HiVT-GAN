# datasets/nuscenes_dataset.py
import os
from typing import Optional, List
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData
from tqdm import tqdm

class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset with proper class filtering and remapping.
    Excludes U-Turns (3) and Off-Map (-1). Maps labels to contiguous 0..5.
    """

    # Classes used in training: 0,1,2,4,5,6 -> mapped to 0..5
    TRAIN_CLASSES = [0, 1, 2, 4, 5, 6]

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

        # Mapping old labels -> new contiguous labels
        self.label_map = {old: new for new, old in enumerate(self.TRAIN_CLASSES)}

        # Directory handling
        standard_path = os.path.join(self.root, self._directory)
        self._processed_dir = standard_path if os.path.isdir(standard_path) else self.root
        self.is_flat = not os.path.isdir(standard_path)

        # List all .pt files
        all_files = sorted(f for f in os.listdir(self._processed_dir) if f.endswith(".pt"))

        # Apply filtering and optional capping
        self._processed_file_names = self._filter_files(all_files, split)

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    def _get_maneuver_id(self, data) -> int:
        str_to_int = {
            "follow": 0, "turn_left": 1, "turn_right": 2,
            "u_turn": 3, "lane_change_left": 4, "lane_change_right": 5,
            "stationary": 6, "off_map": -1
        }
        label = getattr(data, 'maneuver_type', "follow")
        return str_to_int.get(label, 0)

    def _filter_files(self, all_files, split):
        filtered = []
        if split == "train":
            # Limits for capping certain classes
            limits = {0: 500, 6: 300}
            counters = {0: 0, 6: 0}

            for f in tqdm(all_files, desc="Filtering Samples"):
                data = torch.load(os.path.join(self._processed_dir, f))
                m_id = self._get_maneuver_id(data)

                # Exclude U-Turns (3) and Off-Map (-1)
                if m_id == 3 or m_id == -1:
                    continue

                # Keep only valid training classes
                if m_id not in self.TRAIN_CLASSES:
                    continue

                # Apply class capping
                if m_id in limits:
                    if counters[m_id] >= limits[m_id]:
                        continue
                    counters[m_id] += 1

                # Remap label to 0..5
                data.maneuver_id = torch.tensor([self.label_map[m_id]], dtype=torch.long)
                torch.save(data, os.path.join(self._processed_dir, f))
                filtered.append(f)
        else:
            # Validation split: simple slicing
            if self.is_flat:
                num_total = len(all_files)
                num_val = int(num_total * val_ratio)
                num_train = num_total - num_val
                filtered = all_files[num_train:]
            else:
                filtered = all_files
        return filtered

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    def _sanitize(self, data: TemporalData) -> TemporalData:
        # Graph index safety
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

        # Vector fixes
        for key in ["lane_actor_vectors", "lane_vectors"]:
            if hasattr(data, key):
                vec = getattr(data, key)
                if not torch.is_tensor(vec) or vec.numel() == 0 or vec.dim() != 2:
                    setattr(data, key, torch.empty((0, 2), dtype=torch.float))

        # Feature clamping
        for attr in ["turn_directions", "traffic_controls", "is_intersections"]:
            if hasattr(data, attr):
                val = getattr(data, attr)
                setattr(data, attr, torch.clamp(val, 0, 2).long())

        # Ensure maneuver_id exists and is valid
        if not hasattr(data, "maneuver_id") or data.maneuver_id.item() not in range(len(self.TRAIN_CLASSES)):
            m_id = self._get_maneuver_id(data)
            if m_id in self.TRAIN_CLASSES:
                data.maneuver_id = torch.tensor([self.label_map[m_id]], dtype=torch.long)
            else:
                raise ValueError(f"Invalid maneuver_id {m_id} in sample")

        # Ego index
        data.ego_index = torch.tensor([0], dtype=torch.long)

        if data.num_nodes == 0:
            raise ValueError("Invalid sample: graph has zero nodes")

        return data

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
