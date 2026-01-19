import os
import sys
import re
import argparse
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from nuscenes.nuscenes import NuScenes
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import traceback


# ==========================================
# 1. CONFIGURATION & CACHE
# ==========================================
user_name = os.environ.get("USER", "rasoulta")
CACHE_DIR = f"/tmp/{user_name}/fast_cache"

try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except OSError:
    CACHE_DIR = "/mount/studenten/projects/rasoulta/cache_internal"

os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["HF_HOME"] = "/mount/studenten/projects/rasoulta/cache_internal/huggingface"

# ==========================================
# 2. TEMPLATES
# ==========================================
TEMPLATES = {
    "Left Turn": [
        "The ego vehicle executes a left turn.",
        "The vehicle initiates a turn to the left.",
        "A left turn maneuver is performed."
    ],
    "Right Turn": [
        "The ego vehicle executes a right turn.",
        "The vehicle initiates a turn to the right.",
        "A right turn maneuver is performed."
    ],
    "Straight Drive": [
        "The ego vehicle drives straight.",
        "The vehicle proceeds forward without turning.",
        "A straight driving maneuver."
    ],
    "U-Turn": [
        "The ego vehicle performs a U-turn.",
        "The vehicle executes a complete U-turn.",
        "A U-turn maneuver is performed."
    ],
    "Lane Change Left": [
        "The ego vehicle changes lanes to the left.",
        "A lane change to the left lane.",
        "The vehicle merges into the left lane."
    ],
    "Lane Change Right": [
        "The ego vehicle changes lanes to the right.",
        "A lane change to the right lane.",
        "The vehicle merges into the right lane."
    ],
    "Stationary Stop": [
        "The ego vehicle remains stationary.",
        "The vehicle is stopped.",
        "No movement is detected."
    ]
}

LANE_STATUS_TEMPLATES = {
    "maintain": [
        "It maintains its current lane.",
        "The vehicle stays within the lane.",
        "No lane deviation."
    ],
    "change_left": [
        "It is changing lanes to the left.",
        "The vehicle is crossing the left lane divider.",
        "A merge to the left is occurring."
    ],
    "change_right": [
        "It is changing lanes to the right.",
        "The vehicle is crossing the right lane divider.",
        "A merge to the right is occurring."
    ]
}

# ==========================================
# 3. VLM PROMPT
# ==========================================
# Note: We use string concatenation for the markdown backticks to prevent
# breaking the python script display in editors/chat interfaces.
JSON_MARKER = "```json"
END_MARKER = "```"

FULL_PROMPT = f"""
Analyze the driving video and output a single JSON object describing the EGO VEHICLE's behavior and the scene.

### 1. MANEUVER
Choose EXACTLY ONE from: ["Straight Drive", "Left Turn", "Right Turn", "U-Turn", "Lane Change Left", "Lane Change Right", "Stationary Stop"].
- **U-Turn**: A 180-degree turn reversing direction.
- **Lane Change**: A lateral shift between marked lanes.
- **Straight Drive**: Proceeding forward with no turn or lane change.

### 2. LANE STATUS
Choose EXACTLY ONE from: ["maintain", "change_left", "change_right"].
- **maintain**: Staying in the same lane (even while turning).
- **change_left**: Crossing the line to the left.

### 3. SCENE DESCRIPTION
Describe ONLY the environmental context.
- **Include**: Weather (Sunny/Rainy/Night), Road Surface (Dry/Wet), Scene Type (Urban/Highway/Intersection).
- **Negative Constraint**: Do NOT mention the ego vehicle, speed, or other traffic.

### EXAMPLE OUTPUT:
{JSON_MARKER}
{{
  "maneuver": "Left Turn",
  "lane_status": "maintain",
  "scene_description": "It is a sunny day on a dry asphalt intersection with clear lane markings and urban buildings in the background."
}}
{END_MARKER}

### YOUR TASK:
Output the JSON object for this video.
"""

# ==========================================
# 4. PRE-FILTERING (Sample Rates)
# ==========================================
KEEP_RATES = {
    "Straight Drive": 0.15,      # Keep 15%
    "Stationary Stop": 0.20,     # Keep 20%
    "Potential Turn": 1.0,       # Keep 100% of anything interesting
}

def get_rough_category(trajectory):
    """
    Loose Math Filter.
    Returns 'Potential Turn' if there is ANY chance of a turn,
    so we don't accidentally filter out hard samples.
    """
    if trajectory.ndim == 3: trajectory = trajectory[0]
    if len(trajectory) < 6: return "Stationary Stop"

    p0, p_final = trajectory[0, :2], trajectory[-1, :2]
    displacement = float(np.linalg.norm(p_final))
    y_final = float(p_final[1])

    # Hard Physics Check: If moving < 1.0m, it is definitely stationary
    if displacement < 1.0: return "Stationary Stop"

    # Angle Check
    v_start = trajectory[5, :2] - p0
    v_end = p_final - trajectory[-6, :2]
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    diff_deg = np.degrees((angle_end - angle_start + np.pi) % (2 * np.pi) - np.pi)

    # Sensitive Trigger: >15 deg turn OR >1.5m lateral move -> Let VLM decide
    if abs(diff_deg) > 15 or abs(y_final) > 1.5:
        return "Potential Turn"
    
    return "Straight Drive"

# ==========================================
# 5. GENERATOR CLASS
# ==========================================
class CaptionGenerator:
    def __init__(self, dataroot, version, input_dir, output_dir, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.nusc_dataroot = dataroot
        
        print(f"[INIT] Loading NuScenes {version}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        print(f"[INIT] Loading VLM: {model_id}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _collect_future_sequence(self, sample_token):
        tokens = [sample_token]
        cur_token = sample_token
        for _ in range(6):
            try:
                sample = self.nusc.get("sample", cur_token)
                if not sample["next"]: break
                cur_token = sample["next"]
                tokens.append(cur_token)
            except KeyError: break
        return tokens

    def get_image_paths(self, sequence_tokens):
        paths = []
        for token in sequence_tokens:
            try:
                sample = self.nusc.get("sample", token)
                cam_token = sample["data"]["CAM_FRONT"]
                cam_data = self.nusc.get("sample_data", cam_token)
                paths.append(os.path.join(self.nusc_dataroot, cam_data["filename"]))
            except: continue
        return paths

    def generate_full_analysis(self, image_paths):
        """Runs VLM to get Maneuver Class AND Scene Description."""
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if len(valid_paths) < 2: return None

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": valid_paths, "max_pixels": 360 * 420, "fps": 2.0},
                {"type": "text", "text": FULL_PROMPT},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=256, do_sample=False, num_beams=1
                )
        except Exception:
            return None
        
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # Parse JSON
        try:
            cleaned = re.sub(r'```json\s*', '', output_text, flags=re.IGNORECASE)
            cleaned = cleaned.replace('```', '').strip()
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1:
                return json.loads(cleaned[start : end+1])
        except:
            return None
        return None

    def process_all(self, shard_id=0, num_shards=1):
        print(f"[BALANCING] Rates: {KEEP_RATES}")
        
        pt_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".pt"):
                    pt_files.append(os.path.join(root, file))
        
        pt_files.sort()
        random.Random(42).shuffle(pt_files) 
        my_files = pt_files[shard_id::num_shards]
        print(f"[GPU {shard_id}] Scanning {len(my_files)} files...")

        final_stats = {}

        for pt_path in tqdm(my_files, desc=f"GPU {shard_id}"):
            try:
                rel_path = os.path.relpath(pt_path, self.input_dir)
                out_path = os.path.join(self.output_dir, rel_path)
                
                # Check exist
                if os.path.exists(out_path): continue

                data = torch.load(pt_path, weights_only=False)
                traj = data.y if hasattr(data, 'y') else data.get('y')
                if hasattr(traj, 'cpu'): traj = traj.cpu().numpy()

                # 1. ROUGH FILTER
                rough_cat = get_rough_category(traj)
                
                # Apply Sampling Rates
                keep_rate = KEEP_RATES.get(rough_cat, 1.0)
                if random.random() > keep_rate:
                    continue 

                # 2. RUN VLM
                sample_token = os.path.basename(pt_path).replace(".pt", "")
                seq_tokens = self._collect_future_sequence(sample_token)
                image_paths = self.get_image_paths(seq_tokens)
                
                if not image_paths: continue

                vlm_result = self.generate_full_analysis(image_paths)
                if not vlm_result: continue
                
                # 3. EXTRACT & NORMALIZE
                vlm_maneuver = vlm_result.get("maneuver", "Straight Drive")
                vlm_lane = vlm_result.get("lane_status", "maintain")
                scene_desc = vlm_result.get("scene_description", "Standard driving scene.")

                # Sanitize to ensure keys exist in our template dict
                if vlm_maneuver not in TEMPLATES: vlm_maneuver = "Straight Drive"
                if vlm_lane not in LANE_STATUS_TEMPLATES: vlm_lane = "maintain"

                # 4. PICK TEMPLATES
                maneuver_tmpl = random.choice(TEMPLATES[vlm_maneuver])
                lane_tmpl = random.choice(LANE_STATUS_TEMPLATES[vlm_lane])

                # 5. SAVE STRUCTURED DATA
                caption_dict = {
                    "maneuver_type": maneuver_tmpl,
                    "lane_status": lane_tmpl,
                    "scene_description": scene_desc
                }
                
                # Full string for contrastive learning
                full_caption = f"{maneuver_tmpl} {lane_tmpl} {scene_desc}"
                
                # Update Data Object
                data_update = {
                    "caption_dict": caption_dict,
                    "maneuver_category": vlm_maneuver, # Visual Ground Truth
                    "scene_description": full_caption 
                }

                if isinstance(data, dict):
                    data.update(data_update)
                else:
                    for k, v in data_update.items():
                        setattr(data, k, v)
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(data, out_path)
                
                # Stats
                final_stats[vlm_maneuver] = final_stats.get(vlm_maneuver, 0) + 1

            except Exception as e:
                except Exception as e:
                print(f"\n[CRITICAL ERROR] Failed on {pt_path}: {e}")
                traceback.print_exc()
                break # Stop after the first error so we can fix it
        
        print(f"[GPU {shard_id}] Final Distribution: {final_stats}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    gen = CaptionGenerator(
        dataroot=args.dataroot,
        version=args.version,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    gen.process_all(shard_id=args.shard_id, num_shards=args.num_shards)

if __name__ == "__main__":
    main()