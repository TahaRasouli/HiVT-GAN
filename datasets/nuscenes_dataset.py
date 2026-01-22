import os
from typing import Optional, List
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData


class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset with explicit ego indexing.
    Robust against CUDA index crashes.
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

        # 1. Structure Detection
        standard_path = os.path.join(self.root, self._directory)
        if os.path.isdir(standard_path):
            self._processed_dir = standard_path
            self.is_flat = False
        elif os.path.isdir(self.root):
            print(f"[Dataset] Warning: '{self._directory}' not found. Scanning root '{self.root}' directly.")
            self._processed_dir = self.root
            self.is_flat = True
        else:
            raise FileNotFoundError(f"Could not find data in {standard_path} OR {self.root}")

        # 2. File Listing
        all_files = sorted(f for f in os.listdir(self._processed_dir) if f.endswith(".pt"))

        # 3. Split Logic
        if self.is_flat:
            num_total = len(all_files)
            num_val = int(num_total * val_ratio)
            num_train = num_total - num_val

            if split == "train":
                self._processed_file_names = all_files[:num_train]
                print(f"[Dataset] Auto-Split: Assigned {len(self._processed_file_names)} files to TRAIN.")
            elif split == "val":
                self._processed_file_names = all_files[num_train:]
                print(f"[Dataset] Auto-Split: Assigned {len(self._processed_file_names)} files to VAL.")
            else:
                self._processed_file_names = all_files
        else:
            self._processed_file_names = all_files

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    # ------------------------------------------------------------------
    # SANITIZATION LOGIC
    # ------------------------------------------------------------------
    def _sanitize(self, data: TemporalData) -> TemporalData:
        """
        Enforces strict invariants required by HiVT and GPU-safe indexing.
        """

        # --------------------------------------------------------------
        # 1. GRAPH INDEX SANITIZATION
        # --------------------------------------------------------------
        if hasattr(data, "lane_actor_index"):
            lai = data.lane_actor_index
            if not torch.is_tensor(lai):
                data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
            elif lai.numel() == 0:
                data.lane_actor_index = lai.reshape(2, 0)
            elif lai.dim() == 1 and lai.size(0) == 2:
                data.lane_actor_index = lai.reshape(2, 1)
            elif lai.dim() != 2 or lai.size(0) != 2:
                data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)

        if hasattr(data, "edge_index"):
            ei = data.edge_index
            if not torch.is_tensor(ei) or ei.numel() == 0:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
            elif ei.dim() == 1 and ei.size(0) == 2:
                data.edge_index = ei.reshape(2, 1)
            elif ei.dim() != 2 or ei.size(0) != 2:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)

        # --------------------------------------------------------------
        # 2. VECTOR SHAPE FIXES
        # --------------------------------------------------------------
        for key in ["lane_actor_vectors", "lane_vectors"]:
            if hasattr(data, key):
                vec = getattr(data, key)
                if not torch.is_tensor(vec) or vec.numel() == 0 or vec.dim() != 2:
                    setattr(data, key, torch.empty((0, 2), dtype=torch.float))

        # --------------------------------------------------------------
        # 3. MAP FEATURE CLAMPING (EMBEDDING SAFETY)
        # --------------------------------------------------------------
        if hasattr(data, "turn_directions"):
            data.turn_directions = torch.clamp(data.turn_directions, 0, 2).long()

        if hasattr(data, "traffic_controls"):
            data.traffic_controls = torch.clamp(data.traffic_controls, 0, 1).long()

        if hasattr(data, "is_intersections"):
            data.is_intersections = torch.clamp(data.is_intersections, 0, 1).long()

        # --------------------------------------------------------------
        # 4. MANEUVER LABEL EXTRACTION
        # --------------------------------------------------------------
        str_to_int = {
            "Straight Drive": 0, "Go Straight": 0,
            "Left Turn": 1, "Turn Left": 1,
            "Right Turn": 2, "Turn Right": 2,
            "U-Turn": 3, "U Turn": 3,
            "Lane Change Left": 4, "Left Lane Change": 4,
            "Lane Change Right": 5, "Right Lane Change": 5,
            "Stationary": 6, "Stop": 6, "Stationary Stop": 6,
        }

        label_str = None
        if hasattr(data, "maneuver_category"):
            label_str = data.maneuver_category
        elif hasattr(data, "caption_dict"):
            label_str = data.caption_dict.get("category") or data.caption_dict.get("maneuver_category")

        maneuver_id = str_to_int.get(label_str, 0)
        data.maneuver_id = torch.tensor([maneuver_id], dtype=torch.long)

        # --------------------------------------------------------------
        # 5. EXPLICIT EGO INDEX (OPTION A)
        # --------------------------------------------------------------
        if not hasattr(data, "ego_index"):
            # ASSUMPTION: ego node is index 0 (must match preprocessing)
            data.ego_index = torch.tensor([0], dtype=torch.long)
        else:
            data.ego_index = data.ego_index.reshape(1).long()

        # Safety invariant
        assert data.ego_index.item() < data.num_nodes, (
            f"Ego index {data.ego_index.item()} >= num_nodes {data.num_nodes}"
        )

        return data

    # ------------------------------------------------------------------
    # DATASET INTERFACE
    # ------------------------------------------------------------------
    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> TemporalData:
        path = os.path.join(self.processed_dir, self._processed_file_names[idx])
        data = torch.load(path)
        data = self._sanitize(data)
        return data

    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)
