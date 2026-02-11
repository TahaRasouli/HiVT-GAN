import os
import random
import torch
from typing import Optional
from torch_geometric.data import Dataset, Batch
from utils import TemporalData
from tqdm import tqdm


class NuScenesHiVTDataset(Dataset):

    TRAIN_CLASSES = [0,1,2,4,5,6]

    STR_TO_INT = {
        "follow":0,
        "turn_left":1,
        "turn_right":2,
        "u_turn":3,
        "lane_change_left":4,
        "lane_change_right":5,
        "stationary":6,
        "off_map":-1,
    }

    LABEL_MAP = {old:new for new,old in enumerate(TRAIN_CLASSES)}

    def __init__(self, root, split="train", transform=None,
                 max_samples=None, val_ratio=0.1):

        self.root = root
        self.split = split
        self.transform = transform

        all_files = sorted(f for f in os.listdir(root) if f.endswith(".pt"))

        random.seed(42)
        random.shuffle(all_files)

        split_idx = int(len(all_files)*(1-val_ratio))

        if split=="train":
            all_files = all_files[:split_idx]
        else:
            all_files = all_files[split_idx:]

        print(f"[Dataset] Filtering {split} data...")

        limits = {0:500,6:400}
        counters = {0:0,6:0}
        filtered=[]

        for f in tqdm(all_files):
            data = torch.load(os.path.join(root,f),weights_only=False)
            m_id = self._get_maneuver_id(data)

            if m_id not in self.TRAIN_CLASSES:
                continue

            if split=="train" and m_id in limits:
                if counters[m_id]>=limits[m_id]:
                    continue
                counters[m_id]+=1

            filtered.append(f)

        print(f"[Dataset] Final {split} set:",len(filtered))

        self.files=filtered

        if max_samples:
            self.files=self.files[:max_samples]

        super().__init__(root)

    def _get_maneuver_id(self,data):
        label=getattr(data,"maneuver_type","follow")
        return self.STR_TO_INT.get(label,-1)

    def _sanitize(self,data:TemporalData):
        old=self._get_maneuver_id(data)
        new=self.LABEL_MAP[old]
        data.maneuver_id=torch.tensor([new],dtype=torch.long)
        data.ego_index=torch.tensor([0])
        return data

    def len(self):
        return len(self.files)

    def get(self,idx):
        path=os.path.join(self.root,self.files[idx])
        data=torch.load(path,weights_only=False)
        return self._sanitize(data)

    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)
