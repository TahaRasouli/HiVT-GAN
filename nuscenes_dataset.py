import os
from itertools import permutations, product
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data
from utils import TemporalData

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap


class NuScenesHiVTPreprocessor:
    """
    nuScenes (metadata + maps) → HiVT-compatible TemporalData
    Stage-A preprocessing (NO images).
    """

    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        out_dir: str,
        history: int = 20,
        future: int = 30,
        local_radius: float = 50.0,
        lane_resolution: float = 1.0,
    ):
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.history = history
        self.future = future
        self.total_steps = history + future
        self.local_radius = local_radius
        self.lane_resolution = lane_resolution

        self.out_dir = os.path.join(out_dir, f"{split}_processed")
        os.makedirs(self.out_dir, exist_ok=True)

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.scenes = create_splits_scenes()[split]
        self.scene_map = {s["name"]: s for s in self.nusc.scene}
        self.map_cache: Dict[str, NuScenesMap] = {}

        self.sample_tokens = self._build_sample_list()

    # --------------------------------------------------
    def _build_sample_list(self) -> List[str]:
        tokens = []
        for scene_name in self.scenes:
            scene = self.scene_map[scene_name]
            tok = scene["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.nusc.get("sample", tok)["next"] or None
        return tokens

    # --------------------------------------------------
    def _get_map(self, sample_token: str) -> Tuple[str, NuScenesMap]:
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        location = log["location"]

        if location not in self.map_cache:
            self.map_cache[location] = NuScenesMap(
                dataroot=self.dataroot,
                map_name=location,
            )
        return location, self.map_cache[location]

    # --------------------------------------------------
    def process_all(self):
        for token in tqdm(self.sample_tokens, desc=f"Processing {self.split}"):
            out_path = os.path.join(self.out_dir, f"{token}.pt")
            if os.path.exists(out_path):
                continue

            data = self.process_one(token)
            if data is not None:
                torch.save(data, out_path)

    # --------------------------------------------------
    def process_one(self, sample_token: str) -> Optional[TemporalData]:
        sample = self.nusc.get("sample", sample_token)

        tokens = self._collect_sample_sequence(sample)
        if len(tokens) < 2:
            return None

        actor_ids, positions, padding_mask = self._collect_trajectories(tokens)
        if positions is None:
            return None

        num_nodes = positions.size(0)

        # Ego proxy = actor with most valid history
        hist_counts = (~padding_mask[:, :self.history]).sum(dim=1)
        av_index = int(torch.argmax(hist_counts))
        agent_index = av_index

        hist_valid = (~padding_mask[av_index, :self.history]).nonzero().squeeze(-1)
        if hist_valid.numel() < 2:
            return None

        t_last, t_prev = hist_valid[-1], hist_valid[-2]
        origin = positions[av_index, t_last]
        heading = positions[av_index, t_last] - positions[av_index, t_prev]
        theta = torch.atan2(heading[1], heading[0])

        rot = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta),  torch.cos(theta)],
        ], dtype=torch.float32)

        positions_rot = torch.matmul(positions - origin, rot)

        # HiVT rule: no future if unseen at t=19
        padding_mask[padding_mask[:, self.history - 1], self.history:] = True

        rotate_angles = self._compute_rotate_angles(positions_rot, padding_mask)

        # BOS mask (HiVT exact)
        bos_mask = torch.zeros((num_nodes, self.history), dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:self.history] = (
            padding_mask[:, :self.history - 1] &
            ~padding_mask[:, 1:self.history]
        )

        x, positions_abs, y = self._build_motion_features(positions_rot, padding_mask)

        edge_index = torch.LongTensor(
            list(permutations(range(num_nodes), 2))
        ).t().contiguous()

        city, nusc_map = self._get_map(sample_token)

        visible = (~padding_mask[:, self.history - 1]).nonzero().squeeze(-1).tolist()
        pos19 = positions[visible, self.history - 1] if visible else torch.zeros((0, 2))

        lane_data = self._get_lane_features(
            nusc_map, visible, pos19, origin, rot
        )

        return TemporalData(
            x=x,
            positions=positions_abs,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
            padding_mask=padding_mask,
            bos_mask=bos_mask,
            rotate_angles=rotate_angles,
            lane_vectors=lane_data[0],
            is_intersections=lane_data[1],
            turn_directions=lane_data[2],
            traffic_controls=lane_data[3],
            lane_actor_index=lane_data[4],
            lane_actor_vectors=lane_data[5],
            seq_id=sample_token,
            av_index=av_index,
            agent_index=agent_index,
            city=city,
            origin=origin.unsqueeze(0),
            theta=theta,
        )

    # --------------------------------------------------
    def _collect_sample_sequence(self, sample):
        tokens = []
        cur = sample
        for _ in range(self.history):
            tokens.append(cur["token"])
            if not cur["prev"]:
                break
            cur = self.nusc.get("sample", cur["prev"])
        tokens.reverse()

        cur = sample
        for _ in range(self.future):
            if not cur["next"]:
                break
            cur = self.nusc.get("sample", cur["next"])
            tokens.append(cur["token"])
        return tokens

    def _collect_trajectories(self, tokens):
        pos_by_actor = {}
        for t, tok in enumerate(tokens):
            s = self.nusc.get("sample", tok)
            for ann_tok in s["anns"]:
                ann = self.nusc.get("sample_annotation", ann_tok)
                inst = ann["instance_token"]
                pos_by_actor.setdefault(inst, {})[t] = ann["translation"][:2]

        valid = [k for k, v in pos_by_actor.items() if any(t < self.history for t in v)]
        if not valid:
            return [], None, None

        trajs = torch.zeros((len(valid), self.total_steps, 2))
        mask = torch.ones((len(valid), self.total_steps), dtype=torch.bool)

        for i, inst in enumerate(valid):
            for t, pos in pos_by_actor[inst].items():
                trajs[i, t] = torch.tensor(pos)
                mask[i, t] = False

        return valid, trajs, mask

    def _compute_rotate_angles(self, pos, mask):
        angles = torch.zeros(pos.size(0))
        for i in range(pos.size(0)):
            idx = (~mask[i, :self.history]).nonzero().squeeze(-1)
            if idx.numel() >= 2:
                v = pos[i, idx[-1]] - pos[i, idx[-2]]
                angles[i] = torch.atan2(v[1], v[0])
            else:
                mask[i, self.history:] = True
        return angles

    def _build_motion_features(self, pos, mask):
        x = pos.clone()
        N = pos.size(0)

        x[:, self.history:] = torch.where(
            (mask[:, self.history - 1].unsqueeze(-1) | mask[:, self.history:]).unsqueeze(-1),
            torch.zeros(N, self.future, 2),
            x[:, self.history:] - x[:, self.history - 1].unsqueeze(-2)
        )

        x[:, 1:self.history] = torch.where(
            (mask[:, :self.history - 1] | mask[:, 1:self.history]).unsqueeze(-1),
            torch.zeros(N, self.history - 1, 2),
            x[:, 1:self.history] - x[:, :self.history - 1]
        )

        x[:, 0] = 0.0
        return x[:, :self.history], pos.clone(), x[:, self.history:]

    # --------------------------------------------------
    # Correct nuScenes lane API
    # --------------------------------------------------
    def _get_lane_features(self, nusc_map, node_inds, node_pos, origin, rot):
        lane_pos, lane_vec = [], []
        is_inter, turn, control = [], [], []

        lane_ids = set()
        for p in node_pos:
            recs = nusc_map.get_records_in_radius(
                float(p[0]), float(p[1]),
                self.local_radius,
                ["lane", "lane_connector"]
            )
            lane_ids |= set(recs.get("lane", []))
            lane_ids |= set(recs.get("lane_connector", []))

        for lid in lane_ids:
            try:
                pts = nusc_map.discretize_lane(
                    lane_token=lid,
                    resolution_meters=self.lane_resolution
                )
            except Exception:
                continue

            if pts is None or len(pts) < 2:
                continue

            pts = torch.tensor(pts, dtype=torch.float32)[:, :2]
            pts = torch.matmul(pts - origin, rot)
            seg = pts[1:] - pts[:-1]

            lane_pos.append(pts[:-1])
            lane_vec.append(seg)

            count = seg.size(0)
            is_inter.append(torch.zeros(count, dtype=torch.uint8))
            turn.append(torch.zeros(count, dtype=torch.uint8))
            control.append(torch.zeros(count, dtype=torch.uint8))

        if not lane_vec:
            return (
                torch.zeros((0, 2)),
                torch.zeros(0, dtype=torch.uint8),
                torch.zeros(0, dtype=torch.uint8),
                torch.zeros(0, dtype=torch.uint8),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 2)),
            )

        lane_pos = torch.cat(lane_pos)
        lane_vec = torch.cat(lane_vec)
        is_inter = torch.cat(is_inter)
        turn = torch.cat(turn)
        control = torch.cat(control)

        lane_actor_index = torch.LongTensor(
            list(product(range(lane_vec.size(0)), node_inds))
        ).t()

        node_pos = torch.matmul(node_pos - origin, rot)
        lane_actor_vec = (
            lane_pos.repeat_interleave(len(node_inds), 0)
            - node_pos.repeat(lane_vec.size(0), 1)
        )

        mask = torch.norm(lane_actor_vec, dim=1) < self.local_radius

        return (
            lane_vec,
            is_inter,
            turn,
            control,
            lane_actor_index[:, mask],
            lane_actor_vec[mask],
        )


# --------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    proc = NuScenesHiVTPreprocessor(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        out_dir=args.out,
    )
    proc.process_all()


if __name__ == "__main__":
    main()
